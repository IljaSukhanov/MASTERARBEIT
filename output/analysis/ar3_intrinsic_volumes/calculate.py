"""Intrinsic volumes of the full closed AR(3) stability region.

Uses the Euclidean coefficient metric and normal-cycle/Lipschitz--Killing
extension. See derivation.md for the boundary, normalization, and references.
Computes boundary integrals and checks them against independent closed meshes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import dblquad, quad


HERE = Path(__file__).resolve().parent


def boundary_integrals(tolerance: float) -> dict:
    # Saddle: r(s,t) = (s+t, -1-s*t, t), |s| <= 2, |t| <= 1.
    saddle_area, area_error = dblquad(
        lambda s, t: np.sqrt(1 + t*t + (t-s)**2),
        -1, 1, lambda t: -2, lambda t: 2,
        epsabs=tolerance, epsrel=tolerance,
    )
    smooth_mean, smooth_error = dblquad(
        lambda s, t: (1 + s*t) / (1 + t*t + (t-s)**2),
        -1, 1, lambda t: -2, lambda t: 2,
        epsabs=tolerance, epsrel=tolerance,
    )
    # The two edges s=+/-2 are related by (s,t) -> (-s,-t).
    side_angle, side_error = quad(
        lambda t: np.arccos(np.clip(-np.sqrt(3) / np.sqrt(3 + 2*(t-1)**2), -1, 1)),
        -1, 1, epsabs=tolerance, epsrel=tolerance,
    )
    # The two edges t=+/-1 are likewise symmetric.
    end_angle, end_error = quad(
        lambda s: np.arccos(np.clip((-1-s) / np.sqrt(3*(2 + (s-1)**2)), -1, 1)),
        -2, 2, epsabs=tolerance, epsrel=tolerance,
    )
    ridge_mean = np.sqrt(2) * np.arccos(-1/3)
    side_mean = np.sqrt(6) * side_angle
    end_mean = np.sqrt(2) * end_angle
    edge_mean = ridge_mean + side_mean + end_mean
    surface_area = 8*np.sqrt(3) + saddle_area
    return {
        "V0": 1.0,
        "V1": float((smooth_mean + edge_mean) / np.pi),
        "V2": float(surface_area / 2),
        "V3": float(16/3),
        "surface_area": float(surface_area),
        "saddle_area": float(saddle_area),
        "planar_area": float(8*np.sqrt(3)),
        "smooth_integrated_mean_curvature": float(smooth_mean),
        "edge_integrated_mean_curvature": float(edge_mean),
        "edge_contributions": {"ridge": float(ridge_mean),
                               "sides": float(side_mean), "ends": float(end_mean)},
        "quadrature_error_estimates": {
            "saddle_area": float(area_error),
            "smooth_mean": float(smooth_error),
            "side_angle_integral": float(side_error),
            "end_angle_integral": float(end_error),
        },
    }


def closed_mesh(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Conforming graph mesh for the saddle, closed by the two planar faces."""
    ns = 2*n
    s, t = np.meshgrid(np.linspace(-2, 2, ns+1), np.linspace(-1, 1, n+1), indexing="ij")
    vertices = np.column_stack(((s+t).ravel(), (-1-s*t).ravel(), t.ravel()))
    indices = np.arange(len(vertices)).reshape(ns+1, n+1)
    a, b = indices[:-1, :-1].ravel(), indices[1:, :-1].ravel()
    c, d = indices[1:, 1:].ravel(), indices[:-1, 1:].ravel()
    faces = [np.column_stack((a, b, c)), np.column_stack((a, c, d))]
    # B=(1,1,-1), C=(-1,1,1), D=(3,-3,1), A=(-3,-3,-1).
    plus_boundary = np.r_[indices[-1, :], indices[-2::-1, -1]]
    minus_boundary = np.r_[indices[0, :], indices[-1, 0], indices[-2:0:-1, 0]]
    centers = [np.array([1, -1/3, 1/3]), np.array([-1, -1/3, -1/3])]
    normals = [np.array([1, 1, 1]), np.array([-1, 1, -1])]
    for polygon, center, outward in zip([plus_boundary, minus_boundary], centers, normals):
        center_index = len(vertices)
        vertices = np.vstack((vertices, center))
        fan = np.column_stack((np.full(len(polygon), center_index), polygon, np.roll(polygon, -1)))
        points = vertices[fan]
        reverse = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]) @ outward < 0
        fan[reverse] = fan[reverse][:, [0, 2, 1]]
        faces.append(fan)
    return vertices, np.vstack(faces)


def mesh_values(n: int) -> dict:
    vertices, faces = closed_mesh(n)
    points = vertices[faces]
    cross = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    assert np.all(norms > 0)
    normals = cross / norms[:, None]
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    face_ids = np.tile(np.arange(len(faces)), 3)
    unordered = np.sort(directed, axis=1)
    order = np.lexsort((unordered[:, 1], unordered[:, 0]))
    pairs = order.reshape(-1, 2)
    assert np.array_equal(unordered[pairs[:, 0]], unordered[pairs[:, 1]])
    assert len(np.unique(unordered, axis=0)) == len(pairs)
    assert np.array_equal(directed[pairs[:, 0]], directed[pairs[:, 1]][:, ::-1])
    assert len(vertices)-len(pairs)+len(faces) == 2
    first, second = pairs[:, 0], pairs[:, 1]
    e = directed[first]
    tangent = vertices[e[:, 1]]-vertices[e[:, 0]]
    length = np.linalg.norm(tangent, axis=1)
    tangent /= length[:, None]
    n1, n2 = normals[face_ids[first]], normals[face_ids[second]]
    theta = np.arctan2(np.einsum("ij,ij->i", tangent, np.cross(n1, n2)),
                       np.einsum("ij,ij->i", n1, n2))
    mean = float(np.sum(length*theta) / 2)
    volume = float(np.einsum("ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])).sum()/6)
    return {"n": n, "vertices": len(vertices), "triangles": len(faces),
            "boundary_euler_characteristic": 2,
            "V1": mean / np.pi, "V2": float(norms.sum()/4), "V3": volume}


def main() -> None:
    values = boundary_integrals(1e-11)
    coarser = boundary_integrals(1e-8)
    assert max(abs(values[key]-coarser[key]) for key in ("V1", "V2")) < 1e-8
    meshes = [mesh_values(n) for n in (20, 40, 80, 160)]
    for key in ("V1", "V2", "V3"):
        errors = [abs(row[key]-values[key]) for row in meshes]
        assert all(b < a for a, b in zip(errors, errors[1:])), (key, errors)
        assert errors[-1] < 2e-4, (key, errors)
    # Independent volume check: each horizontal section is a triangle.
    slice_volume, _ = quad(lambda t: 4*(1-t*t), -1, 1, epsabs=1e-12)
    assert abs(slice_volume-values["V3"]) < 1e-12
    result = {"convention": "Euclidean coefficients; normal-cycle/Lipschitz--Killing intrinsic volumes",
              "method": "Our boundary-integral derivation; adaptive quadrature; independent closed-mesh checks",
              "values": values, "mesh_convergence": meshes}
    (HERE / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
