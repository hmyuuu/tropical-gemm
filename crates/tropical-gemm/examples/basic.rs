//! Basic example of tropical matrix multiplication.

use tropical_gemm::prelude::*;

fn main() {
    println!("Tropical GEMM - Basic Example\n");
    println!("{}\n", tropical_gemm::version_info());

    // Create two 3x3 matrices
    // A represents edge weights in a graph (rows = from, cols = to)
    // We'll use TropicalMaxPlus for longest path computation
    let a = vec![
        0.0f64, 1.0, 3.0, // from node 0
        2.0, 0.0, 1.0, // from node 1
        1.0, 2.0, 0.0, // from node 2
    ];

    let b = vec![
        0.0f64, 2.0, 1.0, // from node 0
        1.0, 0.0, 3.0, // from node 1
        2.0, 1.0, 0.0, // from node 2
    ];

    println!("Matrix A (3x3):");
    print_matrix(&a, 3, 3);

    println!("\nMatrix B (3x3):");
    print_matrix(&b, 3, 3);

    // Compute C = A ⊗ B using TropicalMaxPlus
    // C[i,j] = max_k(A[i,k] + B[k,j])
    // This gives the maximum 2-hop path weight from i to j
    let c = tropical_matmul::<TropicalMaxPlus<f64>>(&a, 3, 3, &b, 3);

    println!("\nResult C = A ⊗ B (TropicalMaxPlus - longest 2-hop paths):");
    let c_values: Vec<f64> = c.iter().map(|x| x.value()).collect();
    print_matrix(&c_values, 3, 3);

    // Now with argmax tracking
    println!("\n--- With Argmax Tracking ---\n");

    let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(&a, 3, 3, &b, 3);

    println!("Longest 2-hop path analysis:");
    for i in 0..3 {
        for j in 0..3 {
            let value = result.get(i, j).value();
            let via = result.get_argmax(i, j);
            println!("  {} -> {} via {}: total weight = {:.1}", i, j, via, value);
        }
    }

    // MinPlus example for shortest paths
    println!("\n--- Shortest Paths (TropicalMinPlus) ---\n");

    // Use large values instead of infinity for cleaner output
    let dist = vec![
        0.0f64, 1.0, 5.0, // distances from node 0
        2.0, 0.0, 1.0, // distances from node 1
        4.0, 3.0, 0.0, // distances from node 2
    ];

    println!("Distance matrix:");
    print_matrix(&dist, 3, 3);

    let shortest = tropical_matmul::<TropicalMinPlus<f64>>(&dist, 3, 3, &dist, 3);
    let shortest_values: Vec<f64> = shortest.iter().map(|x| x.value()).collect();

    println!("\nShortest 2-hop distances:");
    print_matrix(&shortest_values, 3, 3);

    // AndOr example for reachability
    println!("\n--- Graph Reachability (TropicalAndOr) ---\n");

    let adj = vec![
        false, true, false, // node 0 connects to 1
        false, false, true, // node 1 connects to 2
        true, false, false, // node 2 connects to 0
    ];

    println!("Adjacency matrix:");
    for i in 0..3 {
        print!("  ");
        for j in 0..3 {
            print!("{} ", if adj[i * 3 + j] { "1" } else { "0" });
        }
        println!();
    }

    let adj_tropical: Vec<TropicalAndOr> = adj.iter().map(|&b| TropicalAndOr::new(b)).collect();
    let mut reach = adj_tropical.clone();

    // Manual 2-hop reachability (since AndOr doesn't implement KernelDispatch for matmul)
    // This is a simplified example
    for i in 0..3 {
        for j in 0..3 {
            let mut can_reach = adj_tropical[i * 3 + j];
            for k in 0..3 {
                can_reach = can_reach
                    .tropical_add(adj_tropical[i * 3 + k].tropical_mul(adj_tropical[k * 3 + j]));
            }
            reach[i * 3 + j] = can_reach;
        }
    }

    println!("\n2-hop reachability:");
    for i in 0..3 {
        print!("  ");
        for j in 0..3 {
            print!("{} ", if reach[i * 3 + j].value() { "1" } else { "0" });
        }
        println!();
    }

    println!("\nDone!");
}

fn print_matrix(data: &[f64], rows: usize, cols: usize) {
    for i in 0..rows {
        print!("  ");
        for j in 0..cols {
            print!("{:6.1} ", data[i * cols + j]);
        }
        println!();
    }
}
