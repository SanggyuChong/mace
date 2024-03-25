###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.data import AtomicData
from mace.tools.scatter import scatter_sum

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    NonLinearReadoutBlockLLPR,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    compute_ll_feat_gradients,
)

import copy
from torch.utils.data import DataLoader
# pylint: disable=C0302


@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }


def readout_is_linear(obj: Any):
    if isinstance(obj, torch.jit.RecursiveScriptModule):
        return obj.original_name == "LinearReadoutBlock"
    else:
        return isinstance(obj, LinearReadoutBlock)


def readout_is_nonlinear(obj: Any):
    if isinstance(obj, torch.jit.RecursiveScriptModule):
        return obj.original_name == "NonLinearReadoutBlock"
    else:
        return isinstance(obj, NonLinearReadoutBlock)


@compile_mode("script")
class LLPredRigidityMACE(torch.nn.Module):
    def __init__(
            self,
            model: MACE,
            ll_feat_format: str = "avg",
    ):
        super().__init__()

        # deepcopy the original model
        self.orig_model = copy.deepcopy(model)

        # determine ll_feat size from readout layers
        self.hidden_sizes = []
        self.hidden_size_sum = 0
        for readout in self.orig_model.readouts.children():
            if readout_is_linear(readout):
                cur_size = o3.Irreps(readout.linear.irreps_in)[0].dim
                self.hidden_sizes.append(cur_size)
                self.hidden_size_sum += cur_size
            elif readout_is_nonlinear(readout):
                # wrap modified nonlinear readout block to extract true ll_feat
                # assume only one nonlinear readout in entire MACE architecture
                self.mod_readout = NonLinearReadoutBlockLLPR(readout)
                cur_size = o3.Irreps(readout.linear_2.irreps_in).dim
                self.hidden_sizes.append(cur_size)
                self.hidden_size_sum += cur_size
            else:
                raise TypeError("Unknown readout block type for LLPR at initialization!")

        # initialize (inv_)covariance matrices
        self.register_buffer("covariance",
                             torch.zeros((self.hidden_size_sum, self.hidden_size_sum),
                                         device=next(self.orig_model.parameters()).device
                                         )
                             )
        self.register_buffer("inv_covariance",
                             torch.zeros((self.hidden_size_sum, self.hidden_size_sum),
                                         device=next(self.orig_model.parameters()).device
                                         )
                             )

        # extra params associated with LLPR
        self.ll_feat_format = ll_feat_format
        self.covariance_computed = False
        self.covariance_gradients_computed = False
        self.inv_covariance_computed = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.orig_model.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.orig_model.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.orig_model.spherical_harmonics(vectors)
        edge_feats = self.orig_model.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        ll_feats_list = []
        for i, (interaction, product, readout) in enumerate(zip(
            self.orig_model.interactions.children(),
            self.orig_model.products.children(),
            self.orig_model.readouts.children(),
            )
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            hidden_size = self.hidden_sizes[i]
            # Modified last layer feature pooling for LLPR ----
            # NOTE: ad-hoc solution of checking the readout block type due to
            # to mangling. 1-layer readout is assumed to be LinearReadoutBlock,
            # 3-layer readout is assumed to be NonLinearReadoutBlock
            if torch.jit.is_scripting():
                if len(readout.children()) == 1:
                    node_feats_inv = node_feats[:, :hidden_size]
                    ll_feats_list.append(node_feats_inv)
                    node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
                # 3 layer readout is assumed to be NonLinearReadoutBlock
                elif len(readout.children()) == 3:
                    node_energies, node_feats_after_MLP = self.mod_readout(node_feats)
                    ll_feats_list.append(node_feats_after_MLP[:, :hidden_size])
                    node_energies = node_energies.squeeze(-1)  # [n_nodes, ]
                # throw error when the number of layers does not match above cases
                else:
                    raise TypeError("Unknown readout block type for LLPR at inference!")
            else:
                if readout_is_linear(readout):
                    node_feats_inv = node_feats[:, :hidden_size]
                    ll_feats_list.append(node_feats_inv)
                    node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
                elif readout_is_nonlinear(readout):
                    node_energies, node_feats_after_MLP = self.mod_readout(node_feats)
                    ll_feats_list.append(node_feats_after_MLP[:, :hidden_size])
                    node_energies = node_energies.squeeze(-1)  # [n_nodes, ]
                else:
                    raise TypeError("Unknown readout block type for LLPR at inference!")

            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Aggregate node features
        ll_feats_cat = torch.cat(ll_feats_list, dim=-1)
        ll_feats_agg = scatter_sum(
            src=ll_feats_cat, index=data["batch"], dim=0, dim_size=num_graphs
        )

        if self.ll_feat_format == "sum":
            ll_feats_out = ll_feats_agg

        elif self.ll_feat_format == "avg":
            ll_feats_out = torch.div(ll_feats_agg, num_atoms.unsqueeze(-1))

        elif self.ll_feat_format == "raw":
            ll_feats_out = ll_feats_cat

        else:
            raise RuntimeError("Unsupported last layer feature format!")

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        # return uncertainty if inv_covariance matrix is available
        if self.inv_covariance_computed:
            uncertainty = torch.einsum("ij, jk, ik -> i",
                                       ll_feats_agg,
                                       self.inv_covariance,
                                       ll_feats_agg
                                       )
            uncertainty = uncertainty.unsqueeze(1)
        else:
            uncertainty = None

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "ll_feats": ll_feats_out,
            "uncertainty": uncertainty,
        }

    def compute_covariance(self, train_loader: DataLoader) -> None:
        # Utility function to compute the covariance matrix for a training set.
        for batch in train_loader:
            batch.to(self.covariance.device)
            batch_dict = batch.to_dict()
            output = self.forward(batch_dict)
            ll_feats = output["ll_feats"]
            # Account for the weighting of structures and targets
            cur_weights = torch.mul(batch.weight, batch.energy_weight)
            ll_feats = torch.mul(ll_feats, cur_weights.unsqueeze(-1))
            self.covariance += ll_feats.T @ ll_feats
        self.covariance_computed = True

    def add_gradients_to_covariance(
        self,
        train_loader: DataLoader,
        training: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> None:

        if not self.covariance_computed:
            # Enforce calculation of covariance with energy before considering gradients
            raise RuntimeError("You must first compute the covariance matrix "
                               "with energies before adding the feature "
                               "gradients to the covariance matrix!")

        if self.covariance_gradients_computed:
            # Enforce calculation of covariance with energy before considering gradients
            raise RuntimeError("You have already accounted for gradients in your "
                               "covariance matrix. You are advised to reset the "
                               "covariance and inverse covariance matrices before "
                               "continuing!")

        compute_displacement = compute_virials or compute_stress
        for batch in train_loader:
            batch.to(self.covariance.device)
            batch_dict = batch.to_dict()
            output = self.forward(batch_dict,
                                  training=training,
                                  compute_displacement=compute_displacement,
                                  compute_force=True,
                                  compute_virials=compute_virials,
                                  compute_stress=compute_stress,
                                  )
            ll_feats = output["ll_feats"]

            f_grads, v_grads, s_grads = compute_ll_feat_gradients(
                ll_feats=ll_feats,
                displacement=output["displacement"],
                batch_dict=batch_dict,
                training=training,
                compute_virials=compute_virials,
                compute_stress=compute_stress,
            )

            # Account for the weighting of structures and targets
            f_conf_weights = torch.stack([batch.weight[ii] for ii in batch.batch])
            f_forces_weights = torch.stack([batch.forces_weight[ii] for ii in batch.batch])
            cur_f_weights = torch.mul(f_conf_weights, f_forces_weights)
            f_grads = torch.mul(f_grads, cur_f_weights.view(-1, 1, 1))
            f_grads = f_grads.reshape(-1, ll_feats.shape[-1])
            self.covariance += f_grads.T @ f_grads

            if compute_virials:
                cur_v_weights = torch.mul(batch.weight, batch.virials_weight)
                v_grads = torch.mul(v_grads, cur_v_weights.view(-1, 1, 1, 1))
                v_grads = v_grads.reshape(-1, ll_feats.shape[-1])                
                self.covariance += v_grads.T @ v_grads

            if compute_stress:
                cur_s_weights = torch.mul(batch.weight, batch.stress_weight)
                s_grads = torch.mul(s_grads, cur_s_weights.view(-1, 1, 1, 1))
                s_grads = s_grads.reshape(-1, ll_feats.shape[-1])
                self.covariance += s_grads.T @ s_grads

        self.covariance_gradients_computed = True

    def compute_inv_covariance(self, C: float, sigma: float) -> None:
        # Utility function to set the hyperparameters of the uncertainty model.
        if not self.covariance_computed:
            raise RuntimeError("You must compute the covariance matrix before "
                               "computing the inverse covariance matrix!")
        self.inv_covariance = C * torch.linalg.inv(
            self.covariance + sigma**2 * torch.eye(self.hidden_size_sum, device=self.covariance.device)
            )
        self.inv_covariance_computed = True

    def reset_matrices(self) -> None:
        # Utility function to reset covariance and inv covariance matrices.
        self.covariance = torch.zeros(self.covariance.shape, device=self.covariance.device)
        self.inv_covariance = torch.zeros(self.covariance.shape, device=self.covariance.device)
        self.covariance_computed = False
        self.inv_covariance_computed = False
        self.covariance_gradients_computed = False


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es

        forces, virials, stress = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }

        return output


@compile_mode("script")
class LLPRScaleShiftMACE(torch.nn.Module):
    def __init__(
        self,
        model: ScaleShiftMACE,
        ll_feat_format: str = "avg",
    ):
        super().__init__()

        # deepcopy the original model
        self.orig_model = copy.deepcopy(model)

        # determine ll_feat size from readout layers
        self.hidden_sizes = []
        self.hidden_size_sum = 0
        for readout in self.orig_model.readouts.children():
            if readout_is_linear(readout):
                cur_size = o3.Irreps(readout.linear.irreps_in)[0].dim
                self.hidden_sizes.append(cur_size)
                self.hidden_size_sum += cur_size
            elif readout_is_nonlinear(readout):
                # wrap modified nonlinear readout block to extract true ll_feat
                # assume only one nonlinear readout in entire MACE architecture
                self.mod_readout = NonLinearReadoutBlockLLPR(readout)
                cur_size = o3.Irreps(readout.linear_2.irreps_in).dim
                self.hidden_sizes.append(cur_size)
                self.hidden_size_sum += cur_size
            else:
                raise TypeError("Unknown readout block type for LLPR at initialization!")

        # initialize (inv_)covariance matrices
        self.register_buffer("covariance",
                             torch.zeros((self.hidden_size_sum, self.hidden_size_sum),
                                         device=next(self.orig_model.parameters()).device
                                         )
                             )
        self.register_buffer("inv_covariance",
                             torch.zeros((self.hidden_size_sum, self.hidden_size_sum),
                                         device=next(self.orig_model.parameters()).device
                                         )
                             )

        # extra params associated with LLPR
        self.ll_feat_format = ll_feat_format
        self.covariance_computed = False
        self.covariance_gradients_computed = False
        self.inv_covariance_computed = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.orig_model.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.orig_model.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.orig_model.spherical_harmonics(vectors)
        edge_feats = self.orig_model.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        ll_feats_list = []
        for i, (interaction, product, readout) in enumerate(zip(
            self.orig_model.interactions.children(),
            self.orig_model.products.children(),
            self.orig_model.readouts.children(),
            )
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

            hidden_size = self.hidden_sizes[i]
            # Modified last layer feature pooling for LLPR ----
            # NOTE: ad-hoc solution of checking the readout block type due to
            # to mangling. 1-layer readout is assumed to be LinearReadoutBlock,
            # 3-layer readout is assumed to be NonLinearReadoutBlock
            if torch.jit.is_scripting():
                if len(readout.children()) == 1:
                    node_feats_inv = node_feats[:, :hidden_size]
                    ll_feats_list.append(node_feats_inv)
                # 3 layer readout is assumed to be NonLinearReadoutBlock
                elif len(readout.children()) == 3:
                    _, feat_vec_after_MLP = self.mod_readout(node_feats)
                    ll_feats_list.append(feat_vec_after_MLP[:, :hidden_size])
                # throw error when the number of layers does not match above cases
                else:
                    raise TypeError("Unknown readout block type for LLPR at inference!")
            else:
                if readout_is_linear(readout):
                    node_feats_inv = node_feats[:, :hidden_size]
                    ll_feats_list.append(node_feats_inv)
                elif readout_is_nonlinear(readout):
                    _, node_feats_after_MLP = self.mod_readout(node_feats)
                    ll_feats_list.append(node_feats_after_MLP[:, :hidden_size])
                else:
                    raise TypeError("Unknown readout block type for LLPR at inference!")

        # Aggregate node features
        ll_feats_cat = torch.cat(ll_feats_list, dim=-1)
        ll_feats_agg = scatter_sum(
            src=ll_feats_cat, index=data["batch"], dim=0, dim_size=num_graphs
        )

        if self.ll_feat_format == "sum":
            ll_feats_out = ll_feats_agg

        elif self.ll_feat_format == "avg":
            ll_feats_out = torch.div(ll_feats_agg, num_atoms.unsqueeze(-1))

        elif self.ll_feat_format == "raw":
            ll_feats_out = ll_feats_cat

        else:
            raise RuntimeError("Unsupported last layer feature format!")

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.orig_model.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es

        forces, virials, stress = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        # return uncertainty if inv_covariance matrix is available
        if self.inv_covariance_computed:
            uncertainty = torch.einsum("ij, jk, ik -> i",
                                       ll_feats_agg,
                                       self.inv_covariance,
                                       ll_feats_agg
                                       )
            uncertainty = uncertainty.unsqueeze(1)
        else:
            uncertainty = None

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "ll_feats": ll_feats_out,
            "uncertainty": uncertainty,
        }

        return output

    def compute_covariance(self, train_loader: DataLoader) -> None:
        # Utility function to compute the covariance matrix for a training set.
        for batch in train_loader:
            batch.to(self.covariance.device)
            batch_dict = batch.to_dict()
            output = self.forward(batch_dict)
            ll_feats = output["ll_feats"].detach()
            # Account for the weighting of structures and targets
            cur_weights = torch.mul(batch.weight, batch.energy_weight)
            ll_feats = torch.mul(ll_feats, cur_weights.unsqueeze(-1))
            self.covariance += ll_feats.T @ ll_feats
        self.covariance_computed = True

    def add_gradients_to_covariance(
        self,
        train_loader: DataLoader,
        training: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> None:

        if not self.covariance_computed:
            # Enforce calculation of covariance with energy before considering gradients
            raise RuntimeError("You must first compute the covariance matrix "
                               "with energies before adding the feature "
                               "gradients to the covariance matrix!")

        if self.covariance_gradients_computed:
            # Enforce calculation of covariance with energy before considering gradients
            raise RuntimeError("You have already accounted for gradients in your "
                               "covariance matrix. You are advised to reset the "
                               "covariance and inverse covariance matrices before "
                               "continuing!")

        compute_displacement = compute_virials or compute_stress

        for batch in train_loader:
            batch.to(self.covariance.device)
            batch_dict = batch.to_dict()
            output = self.forward(batch_dict,
                                  training=training,
                                  compute_displacement=compute_displacement,
                                  compute_force=True,
                                  compute_virials=compute_virials,
                                  compute_stress=compute_stress,
                                  )
            ll_feats = output["ll_feats"]

            f_grads, v_grads, s_grads = compute_ll_feat_gradients(
                ll_feats=ll_feats,
                displacement=output["displacement"],
                batch_dict=batch_dict,
                training=training,
                compute_virials=compute_virials,
                compute_stress=compute_stress,
            )

            # Account for the weighting of structures and targets
            f_conf_weights = torch.stack([batch.weight[ii] for ii in batch.batch])
            f_forces_weights = torch.stack([batch.forces_weight[ii] for ii in batch.batch])
            cur_f_weights = torch.mul(f_conf_weights, f_forces_weights)
            f_grads = torch.mul(f_grads, cur_f_weights.view(-1, 1, 1))
            f_grads = f_grads.reshape(-1, ll_feats.shape[-1])
            self.covariance += f_grads.T @ f_grads

            if compute_virials:
                cur_v_weights = torch.mul(batch.weight, batch.virials_weight)
                v_grads = torch.mul(v_grads, cur_v_weights.view(-1, 1, 1, 1))
                v_grads = v_grads.reshape(-1, ll_feats.shape[-1])                
                self.covariance += v_grads.T @ v_grads

            if compute_stress:
                cur_s_weights = torch.mul(batch.weight, batch.stress_weight)
                s_grads = torch.mul(s_grads, cur_s_weights.view(-1, 1, 1, 1))
                s_grads = s_grads.reshape(-1, ll_feats.shape[-1])
                self.covariance += s_grads.T @ s_grads

        self.covariance_gradients_computed = True

    def compute_inv_covariance(self, C: float, sigma: float) -> None:
        # Utility function to set the hyperparameters of the uncertainty model.
        if not self.covariance_computed:
            raise RuntimeError("You must compute the covariance matrix before "
                               "computing the inverse covariance matrix!")
        self.inv_covariance = C * torch.linalg.inv(
            self.covariance + sigma**2 * torch.eye(self.hidden_size_sum, device=self.covariance.device)
            )
        self.inv_covariance_computed = True

    def reset_matrices(self) -> None:
        # Utility function to reset covariance and inv covariance matrices.
        self.covariance = torch.zeros(self.covariance.shape, device=self.covariance.device)
        self.inv_covariance = torch.zeros(self.covariance.shape, device=self.covariance.device)
        self.covariance_computed = False
        self.inv_covariance_computed = False
        self.covariance_gradients_computed = False


class BOTNet(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        gate: Optional[Callable],
        avg_num_neighbors: float,
        atomic_numbers: List[int],
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(inter.irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data.batch, dim=-1, dim_size=data.num_graphs
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        output = {
            "energy": total_energy,
            "contributions": contributions,
            "forces": compute_forces(
                energy=total_energy, positions=data.positions, training=training
            ),
        }

        return output


class ScaleShiftBOTNet(BOTNet):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )

            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            "energy": total_e,
            "forces": compute_forces(
                energy=inter_e, positions=data.positions, training=training
            ),
        }

        return output


@compile_mode("script")
class AtomicDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[
            None
        ],  # Just here to make it compatible with energy models, MUST be None
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        assert atomic_energies is None

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[1]
                )  # Select only l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=True
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the dipoles
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        output = {
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


@compile_mode("script")
class EnergyDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        radial_MLP: Optional[List[int]] = None,
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=False
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_out = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            # node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies = node_out[:, 0]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output
