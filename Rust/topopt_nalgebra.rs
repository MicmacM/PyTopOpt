use std::collections::HashSet;
use nalgebra::{DMatrix, SMatrix, DVector, Matrix, Dyn, Const};
use nalgebra_sparse::{CooMatrix, CscMatrix, factorization::{CscCholesky}};

mod ke;


//TODO Optimisation ideas ?
//? Use hashset for the force SPARSE vector and boolean operation on hashset

//TODO 
//? inspect why the loop isn't changing : problem in the updating of dc or dv value


fn lk() -> SMatrix<f64, 8, 8> {
    let e = 1.0;
    let nu: f64 = 0.3;

    let k = SMatrix::<f64, 8, 1>::from_vec(vec![
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


    let ke = factor * SMatrix::<f64, 8, 8>::from_vec(vec![
        k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
        k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2],
        k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1],
        k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4],
        k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3],
        k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6],
        k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5],
        k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]
    ]);

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

fn prepare_element_stiffness_matrix(e_young: f64, nu: f64) -> SMatrix<f64, 8, 8> {
    // This function should compute the element stiffness matrix KE
    // It uses the Poisson's ratio (nu) to calculate the matrix
    // The calculation involves creating several 4x4 matrices (A11, A12, B11, B12)
    // and then combining them to form the 8x8 KE matrix
    // The resulting KE is a symmetric matrix
    // TODO: Implement the matrix calculations
    //ke::lk(100, e_young, nu)
    lk()
}

fn prepare_fe_analysis(n_x: usize, n_y: usize) -> (DMatrix<usize>, Vec<usize>, Vec<usize>) {
    // TODO Fix the contruction of edof_mat
    let n_node = (n_x + 1) * (n_y + 1);

    let mut edof_mat = DMatrix::<usize>::zeros(n_x * n_y, 8);

    // Populate the edof_mat array
    for elx in 0..n_x {
        for ely in 0..n_y {
            let el = ely + elx * n_y;
            let n1 = (n_y + 1) * elx + ely;
            let n2 = (n_y + 1) * (elx + 1) + ely;
            edof_mat[(el, 0)] = 2 * n1 + 2;
            edof_mat[(el, 1)] = 2 * n1 + 3;
            edof_mat[(el, 2)] = 2 * n2 + 2;
            edof_mat[(el, 3)] = 2 * n2 + 3;
            edof_mat[(el, 4)] = 2 * n2;
            edof_mat[(el, 5)] = 2 * n2 + 1;
            edof_mat[(el, 6)] = 2 * n1;
            edof_mat[(el, 7)] = 2 * n1 + 1;
        }
    }


    let n_el = n_x * n_y;
    let mut i_k = Vec::with_capacity(64 * n_el);
    let mut j_k = Vec::with_capacity(64 * n_el);

    for element in 0..n_el {
        for j in 0..8 {
            for i in 0..8 {
                i_k.push(edof_mat[(element, i)]);
                j_k.push(edof_mat[(element, j)]);
            }
        }
    }

    (edof_mat, i_k, j_k)
}



fn prepare_filter(n_x: usize, n_y: usize, rmin: f64) -> (DMatrix<f64>, DVector<f64>) {
    // This function prepares the density filter:
    // 1. H: Filter matrix
    // 2. Hs: Column sum of H
    // The filter is based on the specified filter radius (rmin)
    // It creates a weight matrix based on the distance between elements
    // TODO: Implement the filter matrix calculation
    (DMatrix::zeros(2 * (n_x + 1) * (n_y + 1), 2 * (n_x + 1) * (n_y + 1)), DVector::zeros(n_x * n_y))
}

fn initialize_design_variables(n_x: usize, n_y: usize, volfrac: f64) -> DMatrix<f64> {
    DMatrix::from_element(n_y, n_x, volfrac)
}

fn compute_sk(ke: &SMatrix<f64, 8, 8>, x_phys: &DMatrix<f64>, emin: f64, emax: f64, penal: f64) -> DVector<f64> {
    let ke_vec = ke.as_slice().to_vec();
    let factor = x_phys.map(|x| emin + x.powf(penal) * (emax - emin));
    let factor_vec = factor.as_slice().to_vec();
    println!("ke_vec_size : {:?}",ke_vec.len());
    println!("factor_vec_size : {:?}",factor_vec.len());
    DVector::from_iterator(ke_vec.len(), ke_vec.iter().zip(factor_vec.iter()).map(|(&k, &f)| k * f))
}


fn select_rows_and_columns(tri_mat: &CooMatrix<f64>, fixed_dofs: &[usize]) -> CooMatrix<f64> {
    let n = tri_mat.ncols();
    let fixed_set: HashSet<usize> = fixed_dofs.iter().cloned().collect();

    // Create a mapping from old indices to new indices using Array1
    let mut index_map: DVector<Option<usize>> = DVector::from_element(n, None);
    let mut new_index = 0;
    for i in 0..n {
        if !fixed_set.contains(&i) {
            index_map[i] = Some(new_index);
            new_index += 1;
        }
    }

    // Create a new TriMat with the selected rows and columns
    let mut new_tri_mat = CooMatrix::new(new_index, new_index);

    for (row, col, &val) in tri_mat.triplet_iter() {
        if let (Some(new_row), Some(new_col)) = (index_map[row], index_map[col]) {
            new_tri_mat.push(new_row, new_col, val);
        }
    }

    new_tri_mat
}

fn select_indices(f: &DVector<f64>, free_dofs: &[usize]) -> DVector<f64> {
    DVector::from_iterator(free_dofs.len(), free_dofs.iter().map(|&i| f[i]))
}


fn solve_system(k: &CooMatrix<f64>, f: &DVector<f64>, free_dofs: &[usize], fixed_dofs: &[usize]) -> DVector<f64> {
    let k_free_coo = select_rows_and_columns(k, free_dofs);
    let f_free = select_indices(f, free_dofs);
    
    let k_free_csc = CscMatrix::from(&k_free_coo);
    
    // Perform Cholesky factorization
    //let k_free_chol = CscCholesky::factor(&k_free_csc).expect("Cholesky factorization failed");
    let k_free_chol = k_free_csc.cholesky().expect("Cholesky failed")
    // Solve the system
    let u_free = k_free_chol.solve(&f_free);
    
    let mut u = DVector::zeros(f.len());
    for (&i, v) in free_dofs.iter().zip(u_free.iter()) {
        u[i] = *v;
    }
    u
}

fn fe_analysis(
    n_x: usize,
    n_y: usize,
    x_phys: &DMatrix<f64>,
    edof_mat: &DMatrix<usize>,
    i_k: &[usize],
    j_k: &[usize],
    f: &DVector<f64>,
    fixed_dofs: &[usize],
    free_dofs: &[usize],
    e0: f64,
    emin: f64,
    penal: f64,
    ke: &SMatrix<f64, 8, 8>,
) -> DVector<f64> {
    let s_k = compute_sk(ke, x_phys, emin, e0, penal);
    let n_dof = 2 * (n_x + 1) * (n_y + 1);

    let mut k = CooMatrix::new(n_dof, n_dof);
    for (&v, (&i, &j)) in s_k.iter().zip(i_k.iter().zip(j_k.iter())) {
        k.push(i, j, v);
    }

    let u = solve_system(&k, f, free_dofs, fixed_dofs);

    u
}


fn compute_ce(
    u: &DVector<f64>,
    edof_mat: &DMatrix<usize>,
    ke: &SMatrix<f64, 8, 8>,
    n_y: usize,
    n_x: usize
) -> DMatrix<f64> {
    let nel = n_y * n_x;
    let mut ce = DVector::zeros(nel);

    for el in 0..nel {
        let ue = DVector::from_iterator(8, (0..8).map(|i| u[edof_mat[(el, i)]]));
        ce[el] = ue.dot(&(ke * &ue));
    }

    DMatrix::from_vec(n_y, n_x, ce.as_slice().to_vec())
}

fn compute_c(x_phys: &DMatrix<f64>, ce: &DMatrix<f64>, emin: f64, e0: f64, penal: f64) -> f64 {
    //! Gros problème la valeur de c est aberemment haute, il faut fix ça
    let material_interpolation = x_phys.map(|x| emin + x.powf(penal) * (e0 - emin));
    /*println!("obj : {:?} ", (&material_interpolation * ce).sum());
    println!("le facteur : {:?}", &material_interpolation);
    println!("le vecteur : {:?}", (&material_interpolation * ce));
    println!("ce : {:?}", &ce);*/
    (material_interpolation * ce).sum()

}

fn compute_dc(x_phys: &DMatrix<f64>, ce: &DMatrix<f64>, emin: f64, e0: f64, penal: f64) -> DMatrix<f64> {
    -penal * (e0 - emin) * x_phys.map(|x| x.powf(penal - 1.0)) * ce
}


fn compute_objective_and_sensitivity(
    x_phys: &DMatrix<f64>,
    u: &DVector<f64>,
    edof_mat: &DMatrix<usize>,
    ke: &SMatrix<f64, 8, 8>,
    e0: f64,
    emin: f64,
    penal: f64,
    n_y : usize,
    n_x : usize,
) -> (f64, DMatrix<f64>, DMatrix<f64>) {
    // Compute the objective function (compliance) and its sensitivity
    // 1. Calculate element-wise strain energy
    // 2. Compute total compliance
    // 3. Compute sensitivity (dc) of the objective w.r.t. design variables
    // TODO: Implement compliance and sensitivity calculations
    let ce = compute_ce(u, edof_mat, &ke, n_y, n_x);
    let c = compute_c(x_phys, &ce, emin, e0, penal);
    let dc = compute_dc(x_phys, &ce, emin, e0, penal);
    let dv = DMatrix::from_element(n_y, n_x, 0.0);

    (c, dc, dv)
}


fn update_design_variables(
    x: &DMatrix<f64>,
    dc: &DMatrix<f64>,
    dv: &DMatrix<f64>,
    volfrac: f64,
    n_x: usize,
    n_y: usize,
) -> DMatrix<f64> {
    let mut l1 = 0.0;
    let mut l2 = 1e9;
    let move_limit = 0.2;
    let mut x_new = x.clone();

    while (l2 - l1) / (l1 + l2) > 1e-3 {
        let l_mid = 0.5 * (l2 + l1);
        
        let indices_iter = (0..n_x).flat_map(|j| (0..n_y).map(move |i| (i, j)));
        for (i, j, &x_val) in indices_iter.zip(x.iter()).map(|((i, j), v)| (i, j, v)) {
            let lower_bound = f64::max(0.0, x_val - move_limit);
            let upper_bound = f64::min(1.0, x_val + move_limit);
            x_new[(i, j)] = f64::max(
                0.0, 
                f64::max(
                    x_val - move_limit,
                    f64::min(
                        1.0,
                        f64::min(
                            x_val + move_limit, 
                            x_val * f64::sqrt(-dc[(i, j)] / (dv[(i, j)] * l_mid))
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

    let change = (&x_new - x).map(f64::abs).iter().cloned().fold(0./0., f64::max);
    println!("Max change in design variables: {}", change);

    x_new
}

fn apply_filter(x: &DMatrix<f64>, h: &DMatrix<f64>, hs: &DVector<f64>) -> DMatrix<f64> {
    let x_flat = x.view(x.shape(), x.shape()).resize(x.len(),1, 0.0);
    let filtered = h.dot(&x_flat);
    let x_phys = &filtered / hs;
    x_phys.into_shape(x.dim()).unwrap()
}

fn compute_change(x: &DMatrix<f64>, x_phys: &DMatrix<f64>) -> f64 {
    // Compute the element-wise absolute difference
    let diff = (x - x_phys).mapv(f64::abs);
    
    // Find the maximum value in the difference array
    diff.iter().fold(0.0, |max, &value| max.max(value))
}


fn print_results(loop_count: usize, c: f64, x_phys: &DMatrix<f64>, change: f64) {
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


fn define_loads_and_supports(n_x: usize, n_y: usize) -> (DVector<f64>, Vec<usize>, Vec<usize>) {    // This function defines the loads and boundary conditions:
    // 1. F: Force vector (sparse in MATLAB, dense Array1 in Rust)
    // 2. fixed_dofs: Indices of fixed degrees of freedom
    // 3. free_dofs: Indices of free degrees of freedom
    // For the half MBB-beam problem:
    // - Apply a downward force at the top left corner
    // - Fix the left edge in x-direction and the bottom right corner in both directions
    // TODO: Implement the force vector and dof vectors
    let n_dof = 2 * (n_x + 1) * (n_y + 1);

    let mut f = DVector::<f64>::from_vec(vec![0.0; n_dof]);
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