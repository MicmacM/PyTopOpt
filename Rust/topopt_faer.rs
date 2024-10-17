use std::collections::{HashSet, HashMap, BTreeSet};
use nalgebra::{DMatrix, SMatrix, DVector, Matrix, Dyn, Const};
use nalgebra_sparse::{CooMatrix, CscMatrix, factorization::{CscCholesky}};
use faer::prelude::*;
use faer::sparse::SparseColMat;
use faer::{Side, Col};


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
    let (f, fixed_dofs, free_dofs) = define_loads_and_supports(n_x, n_y);
    let (edof_mat, i_k, j_k, filter_set) = prepare_fe_analysis(n_x, n_y, &fixed_dofs);
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

        let u = fe_analysis(n_x, n_y, &x_phys, &edof_mat, &i_k, &j_k, &f, &fixed_dofs, &free_dofs, e0, emin, penal, &ke, &filter_set);
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

fn prepare_fe_analysis(n_x: usize, n_y: usize, fixed_dofs: &[usize]) -> (DMatrix<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    // TODO Fix the contruction of edof_mat
    let n_node = (n_x + 1) * (n_y + 1);
    let fixed_set: HashSet<usize> = fixed_dofs.iter().cloned().collect();

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
    let mut filter_set = Vec::new();
    let mut counter = 0;
    fn retard(n : usize, l: &[usize]) -> usize {
        let mut r = 0;
        for e in l {
            if e < &n {
                r += 1
            }
        }
        r
    }

    for element in 0..n_el {
        for j in 0..8 {
            for i in 0..8 {
                if !fixed_set.contains(&edof_mat[(element, i)]) && !fixed_set.contains(&edof_mat[(element, j)]) {
                    i_k.push(edof_mat[(element, i)] - retard(edof_mat[(element, i)], &fixed_dofs));
                    j_k.push(edof_mat[(element, j)] - retard(edof_mat[(element, j)], &fixed_dofs));
                    filter_set.push(counter);
                }
                counter += 1;
            }
        }
    }

    (edof_mat, i_k, j_k, filter_set)
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

fn initialize_design_variables(n_x: usize, n_y: usize, volfrac: f64) -> DVector<f64> {
    DVector::from_element(n_y * n_x, volfrac)
}


fn compute_sk(ke: &SMatrix<f64, 8, 8>, x_phys: &DVector<f64>, emin: f64, emax: f64, penal: f64, i_k: &[usize], j_k: &[usize], filter: &Vec<usize>) -> Vec<f64> {
    
    let ke_row = DVector::from_column_slice(ke.as_slice());
    let factor = x_phys.map(|x| emin + x.powf(penal) * (emax - emin)).transpose();
    let result_dvec = &ke_row * &factor;
    println!("ik : {:?}", &i_k);
    println!("result_dvec : {:?}", &result_dvec);
    println!("filter_set : {:?}", filter);
    let result_vec: Vec<f64> = result_dvec.iter()
        .enumerate()
        .filter(|&(pos, _)| filter.contains(&pos))
        .map(|(_, &value)| value)
        .collect();
    println!("sk : {:?}", &result_vec);
    result_vec
}





fn select_rows_and_columns(tri_mat: &CooMatrix<f64>, fixed_dofs: &[usize]) -> CooMatrix<f64> {
    let n = tri_mat.ncols();
    let fixed_set: HashSet<usize> = fixed_dofs.iter().cloned().collect();

    // Create a mapping from old indices to new indices using Array1
    let mut index_map: DVector<Option<usize>> = DVector::from_element(n, None);
    let mut new_index = 0;
    for i in 0..n {
        //If the index isn't among the fixed dofs, then we add it to index_map
        if !fixed_set.contains(&i) {
            index_map[i] = Some(new_index);
            new_index += 1;
        }
    }

    // Create a new TriMat with the selected rows and columns
    let mut new_tri_mat = CooMatrix::new(new_index, new_index);

    for (row, col, &val) in tri_mat.triplet_iter() {
        // If there are two Some, then we need to keep that value
        if let (Some(new_row), Some(new_col)) = (index_map[row], index_map[col]) {
            new_tri_mat.push(new_row, new_col, val);
        }
    }

    new_tri_mat
}

fn select_indices(f: &Col<f64>, free_dofs: &[usize]) -> faer::Col<f64> {
    /*let mut f_free = Col::zeros(free_dofs.len());
    for (pos, indice_libre) in free_dofs.iter().enumerate() {
        f_free[pos] = f[pos];

    }
    f_free*/
    Col::from_fn(free_dofs.len(), |i| f[free_dofs[i]])
}



fn solve_system(k: &SparseColMat<usize, f64>, f: &Col<f64>, free_dofs: &[usize], fixed_dofs: &[usize]) -> DVector<f64> {
    let f_free = select_indices(f, free_dofs);
    
    
    println!("f_free :{:?}", &f_free);

    // Perform Cholesky factorization
    //let k_free_chol = CscCholesky::factor(&k_free_csc).expect("Cholesky factorization failed");
    let llt = k.sp_cholesky(Side::Lower).unwrap();
    println!("c koi cette merde : {:?}", &llt);
    // Solve the system
    let u_free = llt.solve(&f_free);
    println!("u_free : {:?}", &u_free);
    
    let mut u = DVector::zeros(f.nrows());
    for (&i, v) in free_dofs.iter().zip(u_free.iter()) {
        u[i] = *v;
    }
    u
}

fn fe_analysis(
    n_x: usize,
    n_y: usize,
    x_phys: &DVector<f64>,
    edof_mat: &DMatrix<usize>,
    i_k: &[usize],
    j_k: &[usize],
    f: &Col<f64>,
    fixed_dofs: &[usize],
    free_dofs: &[usize],
    e0: f64,
    emin: f64,
    penal: f64,
    ke: &SMatrix<f64, 8, 8>,
    filter_set: &Vec<usize>,
) -> DVector<f64> {
    let s_k = compute_sk(ke, x_phys, emin, e0, penal, &i_k, &j_k, &filter_set);
    let n_dof = 2 * (n_x + 1) * (n_y + 1);


    //K/2
    let k_vec: Vec<(usize, usize, f64)> = i_k.iter().zip(j_k.iter()).zip(s_k.iter())
        .map(|((i, j), s)| (*i, *j, *s/2.0))
        .collect();

    // K'/2
    let k_vec_transpose: Vec<(usize, usize, f64)> = i_k.iter().zip(j_k.iter()).zip(s_k.iter())
        .map(|((i, j), s)| (*j, *i, *s/2.0))
        .collect();

    // Combine both vectors
    let mut k_sym_2 = k_vec;
    k_sym_2.extend(k_vec_transpose);

    k_sym_2 = combine_tuples(k_sym_2);
    // Sort triplets by ascending i
    k_sym_2.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    
    let mut k_sym = Vec::<(usize, usize, f64)>::new();
    /*
    for chunk in k_sym_2.chunks(2) {
        if let [(i, j, s1), (_, _, s2)] = chunk {
            let new_s = s1 + s2;
            if new_s != 0.0 {
                k_sym.push((*i, *j, new_s));
            }
        }
    }*/
    for (i, j, s) in k_sym_2.iter() {
        if *s != 0.0 {
            k_sym.push((*i, *j, *s))
        }
    }

    fn combine_tuples(tuples: Vec<(usize, usize, f64)>) -> Vec<(usize, usize, f64)> {
        let mut combined: HashMap<(usize, usize), f64> = HashMap::new();
    
        for (i, j, s) in tuples {
            combined.entry((i, j))
                .and_modify(|existing_s| *existing_s += s)
                .or_insert(s);
        }
    
        combined.into_iter()
            .map(|((i, j), s)| (i, j, s))
            .collect()
    }

    

    
    // Display sorted triplets
    
    for (i, j, s) in &k_sym_2 {
        println!("i: {}, j: {}, s: {}", i, j, s);
    }
    
    println!("k_sym len : {:?}", &k_sym.len());
    let mut k = SparseColMat::try_new_from_triplets(free_dofs.len(), free_dofs.len(), &k_sym).unwrap();
    println!("row indices : {:?}", &k.row_indices());
    println!("col pointers : {:?}", &k.col_ptrs());
    //TODO il faut impérativement reprendre les valeurs de u et ne pas juste mettre 0 dans les fixed dofs

    let u = solve_system(&k, f, free_dofs, fixed_dofs);

    u
}


fn compute_ce(
    u: &DVector<f64>,
    edof_mat: &DMatrix<usize>,
    ke: &SMatrix<f64, 8, 8>,
    n_y: usize,
    n_x: usize
) -> DVector<f64> {
    let nel = n_y * n_x;
    let mut ce = DVector::zeros(nel);

    for el in 0..nel {
        let ue = DVector::from_iterator(8, (0..8).map(|i| u[edof_mat[(el, i)]]));
        ce[el] = ue.dot(&(ke * &ue));
    }

    DVector::from_vec(ce.as_slice().to_vec())
}

fn compute_c(x_phys: &DVector<f64>, ce: &DVector<f64>, emin: f64, e0: f64, penal: f64) -> f64 {
    //! Gros problème la valeur de c est aberemment haute, il faut fix ça
    let material_interpolation = x_phys.map(|x| emin + x.powf(penal) * (e0 - emin));
    /*println!("obj : {:?} ", (&material_interpolation * ce).sum());
    println!("le facteur : {:?}", &material_interpolation);
    println!("le vecteur : {:?}", (&material_interpolation * ce));
    println!("ce : {:?}", &ce);*/
    println!("ce : {:?}", &ce);
    println!("material interpotalion : {:?}", &material_interpolation);

    (material_interpolation * ce).sum()

}

fn compute_dc(x_phys: &DVector<f64>, ce: &DVector<f64>, emin: f64, e0: f64, penal: f64) -> DVector<f64> {
    -penal * (e0 - emin) * x_phys.map(|x| x.powf(penal - 1.0)) * ce
}


fn compute_objective_and_sensitivity(
    x_phys: &DVector<f64>,
    u: &DVector<f64>,
    edof_mat: &DMatrix<usize>,
    ke: &SMatrix<f64, 8, 8>,
    e0: f64,
    emin: f64,
    penal: f64,
    n_y : usize,
    n_x : usize,
) -> (f64, DVector<f64>, DVector<f64>) {
    // Compute the objective function (compliance) and its sensitivity
    // 1. Calculate element-wise strain energy
    // 2. Compute total compliance
    // 3. Compute sensitivity (dc) of the objective w.r.t. design variables
    // TODO: Implement compliance and sensitivity calculations
    let ce = compute_ce(u, edof_mat, &ke, n_y, n_x);
    let c = compute_c(x_phys, &ce, emin, e0, penal);
    let dc = compute_dc(x_phys, &ce, emin, e0, penal);
    let dv = DVector::from_element(n_y * n_x, 0.0);

    (c, dc, dv)
}


fn update_design_variables(
    x: &DVector<f64>,
    dc: &DVector<f64>,
    dv: &DVector<f64>,
    volfrac: f64,
    n_x: usize,
    n_y: usize,
) -> DVector<f64> {
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

fn apply_filter(x: &DVector<f64>, h: &DVector<f64>, hs: &DVector<f64>, dc: &DVector<f64>) -> DVector<f64> {
    // Element-wise multiplication of x and dc
    let x_dc = x.component_mul(dc);

    // Element-wise multiplication with h
    let h_x_dc = h.component_mul(&x_dc);

    // Division by hs (element-wise)
    let divided_by_hs = h_x_dc.component_div(hs);

    // Create a vector of 0.001 with the same length as x
    let min_values = DVector::from_element(x.len(), 0.001);

    // Element-wise maximum between 0.001 and x
    let max_values = x.zip_map(&min_values, |a, b| a.max(b));

    // Final element-wise division
    divided_by_hs.component_div(&max_values)
}

fn compute_change(x: &DVector<f64>, x_phys: &DVector<f64>) -> f64 {
    // Compute the element-wise absolute difference
    let diff = (x - x_phys).map(f64::abs);
    
    // Find the maximum value in the difference array
    diff.iter().fold(0.0, |max, &value| max.max(value))
}


fn print_results(loop_count: usize, c: f64, x_phys: &DVector<f64>, change: f64) {
    // Print the results of the current iteration
    // This function is already implemented
    println!(
        " It.:{:5} Obj.:{:11.4} Vol.:{:7.3} ch.:{:7.3}",
        loop_count,
        c,
        x_phys.mean(),
        change
    );
}


fn define_loads_and_supports(n_x: usize, n_y: usize) -> (Col<f64>, Vec<usize>, Vec<usize>) {    // This function defines the loads and boundary conditions:
    // 1. F: Force vector (sparse in MATLAB, dense Array1 in Rust)
    // 2. fixed_dofs: Indices of fixed degrees of freedom
    // 3. free_dofs: Indices of free degrees of freedom
    // For the half MBB-beam problem:
    // - Apply a downward force at the top left corner
    // - Fix the left edge in x-direction and the bottom right corner in both directions
    // TODO: Implement the force vector and dof vectors
    let n_dof = 2 * (n_x + 1) * (n_y + 1);

    //let mut f = DVector::<f64>::from_vec(vec![0.0; n_dof]);
    let mut f = Col::zeros(n_dof);
    f[1] = -1.0;

    // fixeddofs = union([1:2:2*(n_y+1)],[2*(n_x+1)*(n_y+1)]);
    let mut fixed_dofs: HashSet<usize> = (0..2*(n_y+1)).step_by(2).collect();

    // alldofs = [1:2*(n_y+1)*(n_x+1)];
    let all_dofs: HashSet<usize> = (0..n_dof).collect();

    // freedofs = setdiff(alldofs,fixeddofs);
    let mut free_dofs: Vec<usize> = all_dofs.difference(&fixed_dofs).cloned().collect();
    free_dofs.sort();
    
    (f, fixed_dofs.into_iter().collect(), free_dofs)
}

fn main() {
    // Example usage
    optimise(2, 2, 0.5, 3.0, 1.5);
}