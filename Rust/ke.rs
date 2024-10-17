use ndarray::{Array2, array};
extern crate blas_src;
use rayon::prelude::*;

fn b_matrix(xi: f64, eta: f64, w: Option<f64>, l: Option<f64>) -> Array2<f64> {
    let w = w.unwrap_or(1.0);
    let l = l.unwrap_or(1.0);

    let dn1_dx = -w / 2.0 * (1.0 + eta);
    let dn1_dy = l / 2.0 * (1.0 - xi);
    let dn2_dx = -w / 2.0 * (1.0 - eta);
    let dn2_dy = -l / 2.0 * (1.0 - xi);
    let dn3_dx = w / 2.0 * (1.0 + eta);
    let dn3_dy = l / 2.0 * (1.0 + eta);
    let dn4_dx = w / 2.0 * (1.0 - eta);
    let dn4_dy = -l / 2.0 * (1.0 + xi);

    Array2::from_shape_vec((3, 8), vec![
        dn1_dx, 0.0, dn2_dx, 0.0, dn3_dx, 0.0, dn4_dx, 0.0,
        0.0, dn1_dy, 0.0, dn2_dy, 0.0, dn3_dy, 0.0, dn4_dy,
        dn1_dy, dn1_dx, dn2_dy, dn2_dx, dn3_dy, dn3_dx, dn4_dy, dn4_dx,
    ]).unwrap()
}

fn d_matrix(e_young: f64, nu: f64) -> Array2<f64> {
    let f = e_young / (1. - nu.powi(2));
    array![
        [f, nu*f, 0.],
        [nu*f, f, 0.],
        [0., 0., f*(1.-nu)/2.]
    ]
}

fn f(xi: f64, eta: f64, d: &Array2<f64>) -> Array2<f64> {
    let b = b_matrix(xi, eta, None, None);
    b.t().dot(d).dot(&b)
}

pub fn lk(n: i32, e_young: f64, nu: f64) -> Array2<f64> {
    let d = d_matrix(e_young, nu);
    let h: f64 = 2.0 / n as f64;

    let k_matrix: Array2<f64> = (0..n).into_par_iter().map(|i| {
        let mut local_k = Array2::<f64>::zeros((8, 8));
        for j in 0..n {
            let eta: f64 = -1.0 + h * j as f64;
            let xi: f64 = -1.0 + h * i as f64;
            local_k += &f(xi, eta, &d);
            local_k += &f(xi, eta + h, &d);
            local_k += &f(xi + h, eta, &d);
            local_k += &f(xi + h, eta + h, &d);
        }
        local_k
    }).reduce(|| Array2::<f64>::zeros((8, 8)), |a, b| a + b);

    k_matrix * (h.powi(2) / 16.)
}
