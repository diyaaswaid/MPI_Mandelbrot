#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

// Message Tags
#define TAG_TASK 1
#define TAG_RESULT 2
#define TAG_STOP 3

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;

    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;

        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    struct complex c;

    if (size < 2) {
        if (rank == 0)
            printf("Need at least 2 processes (1 master + workers).\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        // MASTER
        int image[HEIGHT][WIDTH];
        int next_row = 0;
        int active_workers = size - 1;
        double start_time, end_time;

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        // Send initial work to each worker
        for (int w = 1; w < size; w++) {
            if (next_row < HEIGHT) {
                MPI_Send(&next_row, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
                next_row++;
            } else {
                int stop = -1;
                MPI_Send(&stop, 1, MPI_INT, w, TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        // Receive results & send new work
        while (active_workers > 0) {
            int row_index;
            int row_data[WIDTH];
            MPI_Status status;

            // receive a completed row
            MPI_Recv(&row_index, 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            MPI_Recv(row_data, WIDTH, MPI_INT, status.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // store it
            for (int j = 0; j < WIDTH; j++)
                image[row_index][j] = row_data[j];

            // give this worker new work
            if (next_row < HEIGHT) {
                MPI_Send(&next_row, 1, MPI_INT, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                next_row++;
            } else {
                int stop = -1;
                MPI_Send(&stop, 1, MPI_INT, status.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();

        save_pgm("mandelbrot_dynamic.pgm", image);
        printf("Dynamic MPI execution time: %f seconds\n", end_time - start_time);
    }

    else {
        // WORKERS
        MPI_Barrier(MPI_COMM_WORLD);

        while (1) {
            int row_index;
            MPI_Status status;

            MPI_Recv(&row_index, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_STOP || row_index < 0)
                break;

            int row_data[WIDTH];

            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (row_index - HEIGHT / 2.0) * 4.0 / HEIGHT;
                row_data[j] = cal_pixel(c);
            }

            MPI_Send(&row_index, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(row_data, WIDTH, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

