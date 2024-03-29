#include <iostream>
#include <mpi.h>

int partition(int array[], int lower_bound, int upper_bound) {
    int pivot = array[upper_bound];
    int i = lower_bound - 1;
    for (int j = lower_bound; j < upper_bound; ++j) {
        if (array[j] <= pivot) {
            i++;
            std::swap(array[i], array[j]);
        }
    }
    std::swap(array[i + 1], array[upper_bound]);
    return i + 1;
}

void quicksort(int array[], int lower_bound, int upper_bound) {
    if (lower_bound < upper_bound) {
        int pivot_index = partition(array, lower_bound, upper_bound);

        // Recursively sort left and right partitions
        quicksort(array, lower_bound, pivot_index - 1);
        quicksort(array, pivot_index + 1, upper_bound);
    }
}

int main(int argc, char** argv) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int input_array[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    int array_size = sizeof(input_array) / sizeof(input_array[0]);

    int chunk_size = array_size / num_procs;
    int* local_array = new int[chunk_size];
    
    MPI_Scatter(input_array, chunk_size, MPI_INT, local_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    quicksort(local_array, 0, chunk_size - 1);

    MPI_Gather(local_array, chunk_size, MPI_INT, input_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    delete[] local_array;

    if (rank == 0) {
        std::cout << "Sorted array:";
        for (int i = 0; i < array_size; ++i) {
            std::cout << " " << input_array[i];
        }
        std::cout << std::endl;
    }
    MPI_Finalize();
    return 0;
}
