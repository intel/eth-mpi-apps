#define BENCHMARK "OSU MPI Multi Latency Test"
/*
 * Copyright (C) 2002-2021 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_util_mpi.h>

char *s_buf, *r_buf;

static int multi_latency(int rank, int pairs);

#ifdef _ENABLE_CUDA_KERNEL_
double measure_kernel_lo(char *, int);
void touch_managed_src(char *, int);
void touch_managed_dst(char *, int);
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

double calculate_total(double, double, double);

int errors_reduced = 0;

int main(int argc, char* argv[])
{
    int rank, nprocs;
    int message_size;
    int pairs;
    int po_ret = 0;
    options.bench = PT2PT;
    options.subtype = LAT;

    set_header(HEADER);
    set_benchmark_name("osu_multi_lat");

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }
    MPI_CHECK(MPI_Init(&argc, &argv));

    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    pairs = nprocs/2;

    if (0 == rank) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(rank);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(rank);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(rank);
                MPI_CHECK(MPI_Finalize());
                break;
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if (rank == 0) {
        print_header(rank, LAT);
        fflush(stdout);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    message_size = multi_latency(rank, pairs);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    if (0 != errors_reduced && options.validate && 0 == rank ) {
        fprintf(stdout, "DATA VALIDATION ERROR: %s exited with status %d on"
                " message size %d.\n", argv[0], EXIT_FAILURE, message_size);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

static int multi_latency(int rank, int pairs)
{
    int size, partner;
    int i, j;
    double t_start = 0.0, t_end = 0.0,
           t_total = 0.0;

    /*needed for the kernel loss calculations*/
    double t_lo=0.0;

    MPI_Status reqstat;

    for (size = options.min_message_size; size <= options.max_message_size;
            size = (size ? size * 2 : 1)) {

        if (allocate_memory_pt2pt_mul_size(&s_buf, &r_buf, rank, pairs, size)) {
            /* Error allocating memory */
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        }

        set_buffer_pt2pt(s_buf, rank, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, rank, options.accel, 'b', size);

        if (size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        } else {
            options.iterations = options.iterations;
            options.skip = options.skip;
        }

#ifdef _ENABLE_CUDA_KERNEL_
        if ((options.src == 'M' && options.MMsrc == 'D') ||
            (options.dst == 'M' && options.MMdst == 'D')) {
            t_lo = measure_kernel_lo(s_buf, size);
        }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        t_total = 0.0;

        for (i = 0; i < options.iterations + options.skip; i++) {
            if (options.validate) {
                set_buffer_validation(s_buf, r_buf, size, options.accel, i);
            }
            for (j = 0; j <= options.warmup_validation; j++) {
                MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
                if (rank < pairs) {
                    partner = rank + pairs;
                    if (i >= options.skip && j == options.warmup_validation) {
                        t_start = MPI_Wtime();
                    }

#ifdef _ENABLE_CUDA_KERNEL_
                    if (options.src == 'M') {
                        touch_managed_src(s_buf, size);
                    }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

                    MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, partner, 1,
                                MPI_COMM_WORLD));
                    MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, partner, 1,
                                MPI_COMM_WORLD, &reqstat));
#ifdef _ENABLE_CUDA_KERNEL_
                    if (options.src == 'M') {
                        touch_managed_src(r_buf, size);
                    }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

                    if (i >= options.skip && j == options.warmup_validation) {
                        t_end = MPI_Wtime();
                        t_total += calculate_total(t_start, t_end, t_lo);
                    }
                } else {
                    partner = rank - pairs;

#ifdef _ENABLE_CUDA_KERNEL_
                    if (options.dst == 'M') {
                        touch_managed_dst(s_buf, size);
                    }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

                    MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, partner, 1,
                                MPI_COMM_WORLD, &reqstat));
#ifdef _ENABLE_CUDA_KERNEL_
                    if (options.dst == 'M') {
                        touch_managed_dst(r_buf, size);
                    }
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

                    MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, partner, 1,
                                MPI_COMM_WORLD));
                }
            }
            if (options.validate) {
                int error = 0, error_temp = 0;
                error = validate_data(r_buf, size, 1, options.accel, i);
                MPI_CHECK(MPI_Reduce(&error, &error_temp, 1, MPI_INT, MPI_SUM,
                            0, MPI_COMM_WORLD));
                errors_reduced += error_temp;
            }
        }

        if (0 == rank) {
            double latency = (t_total * 1e6) / (2.0 * options.iterations);
            if (options.validate) {
                fprintf(stdout, "%-*d%*.*f%*s\n", 10, size, FIELD_WIDTH,
                        FLOAT_PRECISION, latency, FIELD_WIDTH,
                        VALIDATION_STATUS(errors_reduced));
                } else {
                fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                        FLOAT_PRECISION, latency);
                }
            fflush(stdout);
        }

        free_memory_pt2pt_mul(s_buf, r_buf, rank, pairs);

        if (options.validate) {
            MPI_CHECK(MPI_Bcast(&errors_reduced, 1, MPI_INT, 0,
                        MPI_COMM_WORLD));
            if (0 != errors_reduced) {
                break;
            }
        }
    }
return size;
}

#ifdef _ENABLE_CUDA_KERNEL_
double measure_kernel_lo(char *buf, int size)
{
    int i;
    double t_lo = 0.0, t_start, t_end;

    for (i = 0; i < 10; i++) {
        launch_empty_kernel(buf, size);
    }

    for (i = 0; i < 1000; i++) {
        t_start = MPI_Wtime();
        launch_empty_kernel(buf, size);
        synchronize_stream();
        t_end = MPI_Wtime();
        t_lo = t_lo + (t_end - t_start);
    }

    t_lo = t_lo/1000;
    return t_lo;
}

void touch_managed_src(char *buf, int size)
{
    if (options.src == 'M') {
        if (options.MMsrc == 'D') {
            touch_managed(buf, size);
            synchronize_stream();
        } else if ((options.MMsrc == 'H') && size > PREFETCH_THRESHOLD) {
            prefetch_data(buf, size, -1);
            synchronize_stream();
        } else {
            memset(buf, 'c', size);
        }
    }
}

void touch_managed_dst(char *buf, int size)
{
    if (options.dst == 'M') {
        if (options.MMdst == 'D') {
            touch_managed(buf, size);
            synchronize_stream();
        } else if ((options.MMdst == 'H') && size > PREFETCH_THRESHOLD) {
            prefetch_data(buf, size, -1);
            synchronize_stream();
        } else {
            memset(buf, 'c', size);
        }
    }
}
#endif /* #ifdef _ENABLE_CUDA_KERNEL_ */

double calculate_total(double t_start, double t_end, double t_lo)
{
    double t_total;
    if ((options.src == 'M' && options.MMsrc == 'D') && (options.dst == 'M' && options.MMdst == 'D')) {
        t_total = (t_end - t_start) - (2 * t_lo);
    } else if (((options.src == 'M' && options.MMsrc == 'H') &&
                (options.dst == 'M' && options.MMdst == 'D')) ||
               ((options.src == 'M' && options.MMsrc == 'D') &&
                (options.dst == 'M' && options.MMdst == 'H'))) {
        t_total = (t_end - t_start) - t_lo;
    } else if ((options.src == 'M' && options.MMsrc == 'D') ||
                (options.dst == 'M' && options.MMdst == 'D')) {
        t_total = (t_end - t_start) - t_lo;
    } else {
        t_total = (t_end - t_start);
    }

    return t_total;
}
/* vi: set sw=4 sts=4 tw=80: */
