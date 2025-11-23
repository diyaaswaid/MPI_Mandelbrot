#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

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
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = HEIGHT / size;
    int extra = HEIGHT % size;

    int start_row, end_row;

    if (rank < extra) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + (rows_per_proc + 1);
    } else {
        start_row = rank * rows_per_proc + extra;
        end_row = start_row + rows_per_proc;
    }

    int local_rows = end_row - start_row;
    int local_image[local_rows][WIDTH];

    struct complex c;

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for (int i = 0; i < local_rows; i++) {
        int global_i = start_row + i;
        for (int j = 0; j < WIDTH; j++) {
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (global_i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            local_image[i][j] = cal_pixel(c);
        }
    }

    int recvcounts[size];
    int displs[size];

    for (int r = 0; r < size; r++) {
        int r_rows_per_proc = HEIGHT / size;
        int r_extra = HEIGHT % size;
        int r_start, r_end;

        if (r < r_extra) {
            r_start = r * (r_rows_per_proc + 1);
            r_end = r_start + (r_rows_per_proc + 1);
        } else {
            r_start = r * r_rows_per_proc + r_extra;
            r_end = r_start + r_rows_per_proc;
        }

        recvcounts[r] = (r_end - r_start) * WIDTH;
        displs[r] = r_start * WIDTH;
    }

    int (*full_image)[WIDTH] = NULL;
    if (rank == 0) {
        full_image = malloc(sizeof(int) * WIDTH * HEIGHT);
    }

    MPI_Gatherv(
        local_image,
        local_rows * WIDTH,
        MPI_INT,
        full_image,
        recvcounts,
        displs,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        save_pgm("mandelbrot_static.pgm", full_image);
        printf("Static MPI execution time: %f seconds\n", end_time - start_time);
        free(full_image);
    }

    MPI_Finalize();
    return 0;
}
