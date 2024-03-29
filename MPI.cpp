#include <iostream>
#include <mpi.h>

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}


void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        // Recursively sort left and right partitions
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arr[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    int arr_size = sizeof(arr) / sizeof(arr[0]);

    int chunk_size = arr_size / size;
    int* local_arr = new int[chunk_size];
    
    MPI_Scatter(arr, chunk_size, MPI_INT, local_arr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    quicksort(local_arr, 0, chunk_size - 1);

    MPI_Gather(local_arr, chunk_size, MPI_INT, arr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    delete[] local_arr;

    if (rank == 0) {
        std::cout << "Sorted array:";
        for (int i = 0; i < arr_size; ++i) {
            std::cout << " " << arr[i];
        }
        std::cout << std::endl;
    }
    MPI_Finalize();
    return 0;
}