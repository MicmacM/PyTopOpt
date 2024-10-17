use ndarray::{Array, Array1, Array2, s, linalg::kron};
use sprs::{TriMat, TriMatView, CsMat, CsMatView};
use std::collections::HashSet;
use sprs_ldl::*;

mod ke;


//TODO Optimisation ideas ?
//? Use hashset for the force SPARSE vector and boolean operation on hashset

//TODO 
//? inspect why the loop isn't changing : problem in the updating of dc or dv value

fn print_vector(name: &str, vec: &Array1<f64>) {
    println!("{}:", name);
    for (i, &value) in vec.iter().enumerate() {
        println!("{}: {:.6e}", i, value);
    }
    println!();
}

fn print_sparse_matrix(name: &str, mat: CsMatView<f64>) {
    println!("{}:", name);
    for (value, (row, col)) in mat.iter() {
        println!("({}, {}): {:.6e}", row, col, value);
    }
    println!();
}


fn lk() -> Array2<f64> {
    let e = 1.0;
    let nu: f64 = 0.3;

    let k = Array1::from_vec(vec![
        1.0 / 2.0 - nu / 6.0,
        1.0 / 8.0 + nu / 8.0,
        -1.0 / 4.0 - nu / 12.0,
        -1.0 / 8.0 + 3.0 * nu / 8.0,
        -1.0 / 4.0 + nu / 12.0,
        -1.0 / 8.0 - nu / 8.0,
        nu / 6.0,
        1.0 / 8.0 - 3.0 * nu / 8.0
    ]);

    let factor = e / (1.0 - nu.powi(2));

    let ke = factor * Array2::from_shape_vec((8, 8), vec![
        k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
        k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2],
        k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1],
        k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4],
        k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3],
        k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6],
        k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5],
        k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]
    ]).unwrap();

    ke
}



fn optimise(n_x: usize, n_y: usize, volfrac: f64, penal_max: f64, rmin: f64) {
    // ! C'est la FE analysis qui donne une valeur de U aberrante
    let (e0, emin, nu) = initialize_material_properties();
    let ke = prepare_element_stiffness_matrix(e0, nu);
    let (edof_mat, i_k, j_k) = prepare_fe_analysis(n_x, n_y);
    let (f, fixed_dofs, free_dofs) = define_loads_and_supports(n_x, n_y);
    let mut x = initialize_design_variables(n_x, n_y, volfrac);
    let mut x_phys = x.clone();

    let (h, hs) = prepare_filter(n_x, n_y, rmin);
    let mut loop_count = 0;
    let mut change = 1.0;
    let penal = 3.0;

    let mut debug = true;
    // while change > 0.01 {
    while debug {
        loop_count += 1;

        debug = false;

        let u = fe_analysis(n_x, n_y, &x_phys, &edof_mat, &i_k, &j_k, &f, &fixed_dofs, &free_dofs, e0, emin, penal, &ke);
        let (c, dc, dv) = compute_objective_and_sensitivity(&x_phys, &u, &edof_mat, &ke, e0, emin, penal, n_y, n_x);

        //let dc_filtered = apply_filter(&dc, &h, &hs);
        //let dv_filtered = apply_filter(&dv, &h, &hs);

        x = update_design_variables(&x, &dc, &dv, volfrac, n_x, n_y);
        //x_phys = apply_filter(&x, &h, &hs);
        
        change = compute_change(&x, &x_phys);
        
        print_results(loop_count, c, &x, change);
        //x_phys = x.clone();
    }
}

fn initialize_material_properties() -> (f64, f64, f64) {
    // E0: Young's modulus for solid material
    // Emin: Young's modulus for void-like material (very small)
    // nu: Poisson's ratio
    (1.0, 1e-9, 0.3)
}

fn prepare_element_stiffness_matrix(e_young: f64, nu: f64) -> Array2<f64> {
    // This function should compute the element stiffness matrix KE
    // It uses the Poisson's ratio (nu) to calculate the matrix
    // The calculation involves creating several 4x4 matrices (A11, A12, B11, B12)
    // and then combining them to form the 8x8 KE matrix
    // The resulting KE is a symmetric matrix
    // TODO: Implement the matrix calculations
    //ke::lk(100, e_young, nu)
    lk()
}

fn prepare_fe_analysis(n_x: usize, n_y: usize) -> (Array2<usize>, Vec<usize>, Vec<usize>) {
    // TODO Fix the contruction of edof_mat
    let n_node = (n_x + 1) * (n_y + 1);

    let mut edof_mat = Array2::<usize>::zeros((n_x * n_y, 8));

    // Populate the edof_mat array
    for elx in 0..n_x {
        for ely in 0..n_y {
            let el = ely + elx * n_y;
            let n1 = (n_y + 1) * elx + ely;
            let n2 = (n_y + 1) * (elx + 1) + ely;
            edof_mat[[el, 0]] = 2 * n1 + 2;
            edof_mat[[el, 1]] = 2 * n1 + 3;
            edof_mat[[el, 2]] = 2 * n2 + 2;
            edof_mat[[el, 3]] = 2 * n2 + 3;
            edof_mat[[el, 4]] = 2 * n2;
            edof_mat[[el, 5]] = 2 * n2 + 1;
            edof_mat[[el, 6]] = 2 * n1;
            edof_mat[[el, 7]] = 2 * n1 + 1;
        }
    }


    let n_el = n_x * n_y;
    let mut i_k = Vec::with_capacity(64 * n_el);
    let mut j_k = Vec::with_capacity(64 * n_el);

    for element in 0..n_el {
        for j in 0..8 {
            for i in 0..8 {
                i_k.push(edof_mat[[element, i]]);
                j_k.push(edof_mat[[element, j]]);
            }
        }
    }

    (edof_mat, i_k, j_k)
}



fn prepare_filter(n_x: usize, n_y: usize, rmin: f64) -> (Array2<f64>, Array1<f64>) {
    // This function prepares the density filter:
    // 1. H: Filter matrix
    // 2. Hs: Column sum of H
    // The filter is based on the specified filter radius (rmin)
    // It creates a weight matrix based on the distance between elements
    // TODO: Implement the filter matrix calculation
    (Array2::zeros((2 * (n_x + 1) * (n_y + 1), 2 * (n_x + 1) * (n_y + 1))), Array1::zeros(n_x * n_y))
}

fn initialize_design_variables(n_x: usize, n_y: usize, volfrac: f64) -> Array2<f64> {
    Array2::from_elem((n_y, n_x), volfrac)
}

fn compute_sk(ke: &Array2<f64>, x_phys: &Array2<f64>, emin: f64, emax: f64, penal: f64) -> Array1<f64> {
    // KE.flatten()[np.newaxis]).T
    let ke_row = ke.iter().cloned().collect::<Array1<f64>>().insert_axis(ndarray::Axis(0));

    // (Emin + (xPhys)**penal * (Emax-Emin))
    let factor = x_phys.mapv(|x| emin + x.powf(penal) * (emax - emin)).into_shape((x_phys.len(), 1)).expect("Failed to reshape factor");

    // Element-wise multiplication
    println!("ke_vec_size : {:?}",ke_vec.len());
    println!("factor_vec_size : {:?}",factor_vec.len());
    let result = factor.dot(&ke_row);
    // Flatten in column-major (Fortran) order
    //let binding = result.t().to_owned().into_shape(result.len()).expect("Failed to reshape result into 1D array");
    result.iter().cloned().collect()
}


fn select_rows_and_columns(tri_mat: TriMatView<f64>, fixed_dofs: &[usize]) -> CsMat<f64> {
    let n = tri_mat.rows();
    let fixed_set: HashSet<usize> = fixed_dofs.iter().cloned().collect();

    // Create a mapping from old indices to new indices using Array1
    let mut index_map: Array1<Option<usize>> = Array1::from_elem(n, None);
    let mut new_index = 0;
    for i in 0..n {
        if !fixed_set.contains(&i) {
            index_map[i] = Some(new_index);
            new_index += 1;
        }
    }

    // Create a new TriMat with the selected rows and columns
    let mut new_tri_mat = TriMat::new((new_index, new_index));

    for (&val, (row, col)) in tri_mat.triplet_iter() {
        if let (Some(new_row), Some(new_col)) = (index_map[row], index_map[col]) {
            new_tri_mat.add_triplet(new_row, new_col, val);
        }
    }

    // Convert to CsMat and return
    new_tri_mat.to_csr()
}

fn select_indices(f: &Vec<f64>, fixed_dofs: &[usize]) -> Vec<f64> {
    let mut f_free = Vec::new();
    for (index, &value) in f.iter().enumerate() {
        if !fixed_dofs.contains(&index) {
            f_free.push(value);
        }
    }
    f_free
}


fn solve_system(k: TriMatView<f64>, f: &Vec<f64>, free_dofs: &[usize], fixed_dofs :&[usize]) -> Array1<f64> {
    //! FUCKING RANDOM SPRS LDL SOLVER

    // Select the free degrees of freedom 
    let mut k_free = select_rows_and_columns(k, fixed_dofs);
    let f_free: Vec<f64> = select_indices(f, fixed_dofs);

    //Finish the symmetrization of k
    let k_free_transpose = k_free.transpose_view();
    k_free = &k_free + &k_free_transpose;
    k_free.scale(0.5);
    
    println!("k_view {:?}", k_free);
    println!("f_free : {:?}, {:?}", f_free.len(),f_free);
    println!("freedofs {:?}, fixed dofs {:?}", free_dofs, fixed_dofs);
    

    /*
    // Create the LDL factorization
    let ldl = match LdlNumeric::new(k_free.view()) {
        Ok(l) => l,
        Err(e) => {
            println!("LDL factorization failed with error: {:?}", e);
            panic!("LDL factorization failed");
        }
    };

    // Solve the system
    */

    let ldl = Ldl::default();

    let system = ldl.numeric(k_free.view()).unwrap();
    let u_free = system.solve(&f_free);
    
    // Create the full solution vector
    let mut u = Array1::zeros(f.len());
    for (&i, &v) in free_dofs.iter().zip(u_free.iter()) {
        u[i] = v;
    }
    println!("u : {:?}", u);
    u
}

fn fe_analysis(
    n_x: usize,
    n_y: usize,
    x_phys: &Array2<f64>,
    edof_mat: &Array2<usize>,
    i_k: &[usize],
    j_k: &[usize],
    f: &Vec<f64>,
    fixed_dofs: &[usize],
    free_dofs: &[usize],
    e0: f64,
    emin: f64,
    penal: f64,
    ke: &Array2<f64>,
) -> Array1<f64> {
    let s_k = compute_sk(ke, x_phys, emin, e0, penal);
    let n_dof = 2 * (n_x + 1) * (n_y + 1);


    let mut k = TriMat::new((n_dof, n_dof));
    let mut k_transpose = TriMat::new((n_dof, n_dof));

    for (&i, (&j, &v)) in i_k.iter().zip(j_k.iter().zip(s_k.iter())) {
        k.add_triplet(i, j, v);
        k_transpose.add_triplet(j, i, v);
    }

    

    let k_view = k.view();
    let u = solve_system(k_view, f, free_dofs, fixed_dofs);
    u

}


fn compute_ce(u: &Array1<f64>, edof_mat: &Array2<usize>, ke: &Array2<f64>, n_y: usize, n_x: usize) -> Array2<f64> {
    let nel = n_y * n_x;
    let mut ce = Array1::<f64>::zeros(nel);

    for el in 0..nel {
        let ue = Array1::from_vec(edof_mat.row(el).mapv(|i| u[i]).to_vec());
        ce[el] = ue.dot(&ke.dot(&ue));
    }

    ce.into_shape((n_y, n_x)).unwrap()
}

fn compute_c(x_phys: &Array2<f64>, ce: &Array2<f64>, emin: f64, e0: f64, penal: f64) -> f64 {
    //! Gros problème la valeur de c est aberemment haute, il faut fix ça
    let material_interpolation = x_phys.mapv(|x| emin + x.powf(penal) * (e0 - emin));
    /*println!("obj : {:?} ", (&material_interpolation * ce).sum());
    println!("le facteur : {:?}", &material_interpolation);
    println!("le vecteur : {:?}", (&material_interpolation * ce));
    println!("ce : {:?}", &ce);*/
    (material_interpolation * ce).sum()

}

fn compute_dc(x_phys: &Array2<f64>, ce: &Array2<f64>, emin: f64, e0: f64, penal: f64) -> Array2<f64> {
    -penal * (e0 - emin) * x_phys.mapv(|x| x.powf(penal - 1.0)) * ce
}


fn compute_objective_and_sensitivity(
    x_phys: &Array2<f64>,
    u: &Array1<f64>,
    edof_mat: &Array2<usize>,
    ke: &Array2<f64>,
    e0: f64,
    emin: f64,
    penal: f64,
    n_y : usize,
    n_x : usize,
) -> (f64, Array2<f64>, Array2<f64>) {
    // Compute the objective function (compliance) and its sensitivity
    // 1. Calculate element-wise strain energy
    // 2. Compute total compliance
    // 3. Compute sensitivity (dc) of the objective w.r.t. design variables
    // TODO: Implement compliance and sensitivity calculations
    let ce = compute_ce(u, edof_mat, &ke, n_y, n_x);
    let c = compute_c(x_phys, &ce, emin, e0, penal);
    let dc = compute_dc(x_phys, &ce, emin, e0, penal);
    let dv = Array2::ones((n_y, n_x));

    (c, dc, dv)
}


fn update_design_variables(
    x: &Array2<f64>,
    dc: &Array2<f64>,
    dv: &Array2<f64>,
    volfrac: f64,
    n_x: usize,
    n_y: usize,
) -> Array2<f64> {
    let mut l1 = 0.0;
    let mut l2 = 1e9;
    let move_limit = 0.2;
    let mut x_new = x.clone();

    while (l2 - l1) / (l1 + l2) > 1e-3 {
        let l_mid = 0.5 * (l2 + l1);
        
        for ((i, j), &x_val) in x.indexed_iter() {
            let lower_bound = f64::max(0.0, x_val - move_limit);
            let upper_bound = f64::min(1.0, x_val + move_limit);
            x_new[[i, j]] = f64::max(
                0.0, 
                f64::max(
                    x_val - move_limit,
                    f64::min(
                        1.0,
                        f64::min(
                            x_val + move_limit, 
                            x_val * f64::sqrt(-dc[[i, j]] / (dv[[i, j]] * l_mid))
                        )
                    )
                )
            );
        }

        if x_new.sum() > volfrac * (n_x * n_y) as f64 {
            l1 = l_mid;
        } else {
            l2 = l_mid;
        }
    }

    let change = (&x_new - x).mapv(f64::abs).iter().cloned().fold(0./0., f64::max);
    println!("Max change in design variables: {}", change);

    x_new
}

fn apply_filter(x: &Array2<f64>, h: &Array2<f64>, hs: &Array1<f64>) -> Array2<f64> {
    let x_flat = x.view().into_shape((x.len(),)).unwrap();
    let filtered = h.dot(&x_flat);
    let x_phys = &filtered / hs;
    x_phys.into_shape(x.dim()).unwrap()
}

fn compute_change(x: &Array2<f64>, x_phys: &Array2<f64>) -> f64 {
    // Compute the element-wise absolute difference
    let diff = (x - x_phys).mapv(f64::abs);
    
    // Find the maximum value in the difference array
    diff.iter().fold(0.0, |max, &value| max.max(value))
}


fn print_results(loop_count: usize, c: f64, x_phys: &Array2<f64>, change: f64) {
    // Print the results of the current iteration
    // This function is already implemented
    println!(
        " It.:{:5} Obj.:{:11.4} Vol.:{:7.3} ch.:{:7.3}",
        loop_count,
        c,
        x_phys.mean().unwrap(),
        change
    );
}


fn define_loads_and_supports(n_x: usize, n_y: usize) -> (Vec<f64>, Vec<usize>, Vec<usize>) {    // This function defines the loads and boundary conditions:
    // 1. F: Force vector (sparse in MATLAB, dense Array1 in Rust)
    // 2. fixed_dofs: Indices of fixed degrees of freedom
    // 3. free_dofs: Indices of free degrees of freedom
    // For the half MBB-beam problem:
    // - Apply a downward force at the top left corner
    // - Fix the left edge in x-direction and the bottom right corner in both directions
    // TODO: Implement the force vector and dof vectors
    let n_dof = 2 * (n_x + 1) * (n_y + 1);

    let mut f = vec![0.0; n_dof];
    f[1] = -1.0;

    // fixeddofs = union([1:2:2*(n_y+1)],[2*(n_x+1)*(n_y+1)]);
    let mut fixed_dofs: HashSet<usize> = (0..2*(n_y+1)).step_by(2).collect();
    fixed_dofs.insert(n_dof - 1);

    // alldofs = [1:2*(n_y+1)*(n_x+1)];
    let all_dofs: HashSet<usize> = (0..n_dof).collect();

    // freedofs = setdiff(alldofs,fixeddofs);
    let free_dofs: Vec<usize> = all_dofs.difference(&fixed_dofs).cloned().collect();

    (f, fixed_dofs.into_iter().collect(), free_dofs)
}

fn main() {
    // Example usage
    optimise(2, 1, 0.5, 3.0, 1.5);
}