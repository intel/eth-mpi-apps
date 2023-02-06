/* 
  Copyright (c) 2017, Intel Corporation

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of Intel Corporation nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#ifndef __USE_XOPEN2K
#define __USE_XOPEN2K 1		/* for posix_memalign() on RHEL 4 */
#endif
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>
#include <zlib.h>

#ifdef MPI_STRESS_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef MPI_STRESS_ONEAPI
#include <level_zero/ze_api.h>
#endif // MPI_STRESS_ONEAPI

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

#ifdef MPI_STRESS_CUDA
#define CUDA_CALL(func, args...) do {					\
		cudaError_t cudaerr;					\
		cudaerr = func(args);					\
		if (cudaerr != cudaSuccess) {				\
			fprintf(stderr,					\
				"CUDA failure: %s() (at %s:%d)"		\
				"returned %d: %s\n",			\
				#func, __FILE__, __LINE__, cudaerr,	\
				cudaGetErrorString(cudaerr));		\
			sleep(2);					\
			exit(cudaerr);					\
		}							\
	} while (0)
#define MAX_DEVICES          8
#endif

#ifdef MPI_STRESS_ONEAPI

#define MAX_ZE_DEVICES       8

struct ze_dev_ctxt {
	ze_device_handle_t dev;
	uint32_t ordinal; /* CmdQGrp ordinal for the 1st copy_only engine */
	uint32_t index;   /* Cmdqueue index within the CmdQGrp */
	uint32_t num_queues; /* Number of queues in the CmdQGrp */
	ze_command_queue_handle_t cq;
	ze_command_list_handle_t cl;
};

static ze_context_handle_t ze_context = NULL;
static ze_driver_handle_t ze_driver = NULL;
static ze_device_handle_t devices[MAX_ZE_DEVICES];
static struct ze_dev_ctxt ze_devices[MAX_ZE_DEVICES];
static struct ze_dev_ctxt *cur_ze_dev = NULL;
static ze_command_queue_handle_t ze_cq = NULL;
static ze_command_list_handle_t ze_cl = NULL;

#define MPI_STRESS_ONEAPI_ZE_CALL(func, args...) do { \
  ze_result_t result; \
  result = func(args);	\
  if(result != ZE_RESULT_SUCCESS) { \
    fprintf(stderr, "OneAPI Level Zero failure: %s() (at %s:%d) " \
      "returned %d: %s\n", \
      #func, __FILE__, __LINE__, result, psmi_oneapi_ze_result_to_string(result)); \
    sleep(2);	\
    exit(result); \
  } \
} while (0)

static const char* psmi_oneapi_ze_result_to_string(const ze_result_t result) {
#define ZE_RESULT_CASE(RES) case ZE_RESULT_##RES: return #RES

  switch (result) {
  ZE_RESULT_CASE(SUCCESS);
  ZE_RESULT_CASE(NOT_READY);
  ZE_RESULT_CASE(ERROR_UNINITIALIZED);
  ZE_RESULT_CASE(ERROR_DEVICE_LOST);
  ZE_RESULT_CASE(ERROR_INVALID_ARGUMENT);
  ZE_RESULT_CASE(ERROR_OUT_OF_HOST_MEMORY);
  ZE_RESULT_CASE(ERROR_OUT_OF_DEVICE_MEMORY);
  ZE_RESULT_CASE(ERROR_MODULE_BUILD_FAILURE);
  ZE_RESULT_CASE(ERROR_INSUFFICIENT_PERMISSIONS);
  ZE_RESULT_CASE(ERROR_NOT_AVAILABLE);
  ZE_RESULT_CASE(ERROR_UNSUPPORTED_VERSION);
  ZE_RESULT_CASE(ERROR_UNSUPPORTED_FEATURE);
  ZE_RESULT_CASE(ERROR_INVALID_NULL_HANDLE);
  ZE_RESULT_CASE(ERROR_HANDLE_OBJECT_IN_USE);
  ZE_RESULT_CASE(ERROR_INVALID_NULL_POINTER);
  ZE_RESULT_CASE(ERROR_INVALID_SIZE);
  ZE_RESULT_CASE(ERROR_UNSUPPORTED_SIZE);
  ZE_RESULT_CASE(ERROR_UNSUPPORTED_ALIGNMENT);
  ZE_RESULT_CASE(ERROR_INVALID_SYNCHRONIZATION_OBJECT);
  ZE_RESULT_CASE(ERROR_INVALID_ENUMERATION);
  ZE_RESULT_CASE(ERROR_UNSUPPORTED_ENUMERATION);
  ZE_RESULT_CASE(ERROR_UNSUPPORTED_IMAGE_FORMAT);
  ZE_RESULT_CASE(ERROR_INVALID_NATIVE_BINARY);
  ZE_RESULT_CASE(ERROR_INVALID_GLOBAL_NAME);
  ZE_RESULT_CASE(ERROR_INVALID_KERNEL_NAME);
  ZE_RESULT_CASE(ERROR_INVALID_FUNCTION_NAME);
  ZE_RESULT_CASE(ERROR_INVALID_GROUP_SIZE_DIMENSION);
  ZE_RESULT_CASE(ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
  ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
  ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
  ZE_RESULT_CASE(ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
  ZE_RESULT_CASE(ERROR_INVALID_COMMAND_LIST_TYPE);
  ZE_RESULT_CASE(ERROR_OVERLAPPING_REGIONS);
  ZE_RESULT_CASE(ERROR_UNKNOWN);
  default:
    return "Unknown error";
  }

#undef ZE_RESULT_CASE
}

static void *mycalloc (size_t nmemb, size_t size);
static void myfree (void *ptr);
static void oneapi_ze_find_copy_only_engine(ze_device_handle_t dev,
					      struct ze_dev_ctxt *ctxt)
{
	uint32_t count = 0;
	ze_command_queue_group_properties_t *props = NULL;
	int i;

	/* Set the default */
	ctxt->ordinal = 0;
	ctxt->index = 0;
	ctxt->num_queues = 1;
	MPI_STRESS_ONEAPI_ZE_CALL(zeDeviceGetCommandQueueGroupProperties, dev,
			    &count, NULL);
	props = mycalloc(count, sizeof(*props));
	if (!props) {
		fprintf(stderr, "Failed to allocate mem for CmdQ Grp\n");
		return;
	}
	MPI_STRESS_ONEAPI_ZE_CALL(zeDeviceGetCommandQueueGroupProperties, dev,
			    &count, props);

	/* Select the first copy-only engine group if possible */
	for (i = count - 1; i >= 0; i--) {
		if ((props[i].flags &
		    ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
		    !(props[i].flags &
		      ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
			ctxt->ordinal = i;
			ctxt->num_queues = props[i].numQueues;
			break;
		}
	}
	myfree(props);
}

static void oneapi_ze_malloc(void **pptr, size_t size)
{
  size_t alignment = 64;
  ze_device_mem_alloc_desc_t dev_desc = {
    .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
    .flags = 0,
    .ordinal = 0
  };
  MPI_STRESS_ONEAPI_ZE_CALL(zeMemAllocDevice, ze_context, &dev_desc, size, alignment, cur_ze_dev->dev, pptr);
}

static void oneapi_ze_memcpy(void *dstptr, const void *srcptr, size_t size) {
  MPI_STRESS_ONEAPI_ZE_CALL(zeCommandListAppendMemoryCopy, ze_cl, dstptr, srcptr, size, NULL, 0, NULL);
  MPI_STRESS_ONEAPI_ZE_CALL(zeCommandListClose, ze_cl);
  MPI_STRESS_ONEAPI_ZE_CALL(zeCommandQueueExecuteCommandLists, ze_cq, 1, &ze_cl, NULL);
  MPI_STRESS_ONEAPI_ZE_CALL(zeCommandQueueSynchronize, ze_cq, UINT32_MAX);
  MPI_STRESS_ONEAPI_ZE_CALL(zeCommandListReset, ze_cl);
}

static void oneapi_ze_cmd_create(ze_device_handle_t dev, struct ze_dev_ctxt *ctxt)
{
	ze_command_queue_desc_t ze_cq_desc = {
		.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
		.flags = 0,
		.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT,
		.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
		.ordinal = 0 /* this must be less than device_properties.numAsyncComputeEngines */
	};
	ze_command_list_desc_t ze_cl_desc = {
		.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
		.flags = 0
	};

	oneapi_ze_find_copy_only_engine(dev, ctxt);
	ze_cq_desc.ordinal = ctxt->ordinal;
	ze_cq_desc.index = ctxt->index;

	MPI_STRESS_ONEAPI_ZE_CALL(zeCommandQueueCreate, ze_context, dev,
			    &ze_cq_desc, &ctxt->cq);

	ze_cl_desc.commandQueueGroupOrdinal = ctxt->ordinal;
	MPI_STRESS_ONEAPI_ZE_CALL(zeCommandListCreate, ze_context, dev, &ze_cl_desc,
			    &ctxt->cl);
	ctxt->dev = dev;
}
#endif // MPI_STRESS_ONEAPI

#define DEFAULT_INITIAL_BYTE 0x00

static const uint32_t RANDOM_SIZE = 0xFFFFFFFF;	/* marker for random sizes */

char myhostname[MPI_MAX_PROCESSOR_NAME];

typedef
int (*mpi_send_fn_t)(const void *buf, int count, MPI_Datatype dtype, int dest,
		     int tag, MPI_Comm comm, MPI_Request *request);

static void fatal (char *str, ...)
{
  va_list ap;
  va_start(ap, str);
  fflush(stdout);
  fprintf(stderr, "Error (%s): ", myhostname);
  vfprintf(stderr, str, ap);
  fflush(stderr);
  va_end(ap);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  abort();
}

static void *mycalloc (size_t nmemb, size_t size)
{
  void *result = calloc(nmemb, size);
  if (result == NULL) {
    fatal("Out of heap\n");
  }
  return result;
}

static void *mymalloc (size_t size)
{
  void *result = malloc(size);
  if (result == NULL) {
    fatal("Out of heap\n");
  }
  return result;
}

static void *mymemalign (size_t size, size_t align)
{
  void *result;
  if (align == 0) {
    result = malloc(size);
    if (result == NULL) {
      fatal("Out of heap\n");
    }
  }
  else {
    int err = posix_memalign(&result, align, size);
    if (err == EINVAL) {
      fatal("Bad memory alignment value\n");
    }
    else if (err == ENOMEM) {
      fatal("Out of heap\n");
    }
    else if (err != 0) {
	    fatal("Memory allocation failure, error code %d\n", err);
    }  
  }
  return result;
}

static void myfree (void *ptr)
{
  free(ptr);
}

static long int mystrtol (const char *nptr, char **endptr, int base)
{
  int e = errno;
  long int l;
  errno = 0;
  l = strtol(nptr, endptr, base);
  if (errno) {
    fatal("bad numeric conversion in strtol\n");
  }
  errno = e;
  return l;
}

static unsigned long int mystrtoul (const char *nptr, char **endptr, int base)
{
  int e = errno;
  unsigned long int ul;
  errno = 0;
  ul = strtoul(nptr, endptr, base);
  if (errno) {
    fatal("bad numeric conversion in strtoul\n");
  }
  errno = e;
  return ul;
}

static int myrand_r (unsigned *seedp, unsigned min, unsigned max)
{
  unsigned range = max - min + 1;
  if (range == 0) {
    return min;
  }
  else {
    unsigned r = range;
    unsigned n = rand_r(seedp);
    int numbits = 1;
    assert(max >= min);
    while (r) {
      numbits++;
      r >>= 1;
    }
    return min + ((n >> (32 - numbits)) % range);
  }
}

static size_t ceilpow2 (size_t n)
{
  size_t r = 1;
  while (r < n) {
    r <<= 1;
  }
  return r;
}

enum test_type
{
  STANDARD,
  SMALL_ONE_BYTE,
  SMALL_TWO_BYTE,
  SMALL_MULTI_BYTE
};

typedef uint16_t small_msg_hdr_t;

typedef union small_msg {
  uint8_t data_one_byte;
  uint16_t data_two_byte;
  small_msg_hdr_t data_multi_byte_chk;
} small_msg_t;

typedef struct msg_hdr
{
  uint32_t src;
  uint32_t dst;
  uint32_t tag;
  uint32_t len;
  uint32_t seq;
  uint32_t hdr_chk;
  uint32_t data_chk;
  uint32_t seed;
} msg_hdr_t;

typedef struct msg_state
{
  uint8_t *buf;
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
  uint8_t *gpubuf;
#endif

  MPI_Request req;
  uint32_t poison_seed;
} msg_state_t;

typedef struct msg_array
{
  uint32_t count;
  uint32_t size;
  uint32_t index;
  msg_state_t *msgs;
} msg_array_t;

typedef struct peer_state
{
  uint32_t send_seq;
  uint32_t recv_seq;
  int completed;
} peer_state_t;

typedef struct params
{
  int verbose;
  int progress;
  int show_options;
  int poison;
  int zero;
  int quiet;
  int include_self;
  int include_shm;
  int use_crc_check;
  int use_xor_check;
  int data_check;
  int use_random_data;
  int use_misaligned_data;
  int round_robin;
  int grid_ndims;
  int periodic_grid;
  int uni_grid;
  int use_random_length;
  int use_small_messages;
  uint32_t repeats;
  uint32_t runtime_mins;
  uint32_t min_msg_size;
  uint32_t incr_msg_size;
  uint32_t max_msg_size;
  uint32_t min_msgs;
  uint32_t max_msgs;
  uint32_t max_data;
  uint32_t window_size;
  uint32_t align;
  uint8_t  initial_byte;
  mpi_send_fn_t mpi_send_fn;
  char *mpi_send_fn_name;
  int max_errors;
  uint32_t wait_on_start;
  uint32_t wait_on_exit;
  int use_gpu_send;
  int use_gpu_recv;
} params_t;

typedef struct local_state
{
  params_t params;
  unsigned seed;
  peer_state_t *peers;
  uint32_t comm_rank;
  uint32_t comm_size;
  char *hostnames;
  int *grid_dims;
  uint32_t send_size;
  uint32_t *send_mapping;
  uint32_t send_round_robin;
  uint32_t recv_size;
  uint32_t min_tag;
  uint32_t max_tag;
  uint32_t completed;
  uint8_t *recv_validate_buf;
  int test_mode;
} local_state_t;

static const char *get_hostname (local_state_t *local, uint32_t rank)
{
  if (rank < local->comm_size)
    return local->hostnames + (MPI_MAX_PROCESSOR_NAME * rank);
  else
    return "bad rank";
}

static uint16_t calculate_xor_16 (uint32_t len, uint8_t *data, uint16_t xor)
{
  uint32_t i;
  uint16_t *data16 = (uint16_t *) data;
  uint32_t len32 = len >> 2;
  for (i = 0; i < len32; i++) {
    xor = xor ^ data16[i];
  }
  len = len & 0x3;
  data += len32 << 2;
  for (i = 0; i < len; i++) {
    xor = xor ^ (data[i] << (i << 3));
  }
  return xor;
}

static uint32_t calculate_xor_32 (uint32_t len, uint8_t *data, uint32_t xor)
{
  uint32_t i;
  uint32_t *data32 = (uint32_t *) data;
  uint32_t len32 = len >> 2;
  for (i = 0; i < len32; i++) {
    xor = xor ^ data32[i];
  }
  len = len & 0x3;
  data += len32 << 2;
  for (i = 0; i < len; i++) {
    xor = xor ^ (data[i] << (i << 3));
  }
  return xor;
}

static uint32_t calculate_xor_64 (uint32_t len, uint8_t *data, uint32_t xor)
{
  uint32_t i;
  uint64_t *data64 = (uint64_t *) data;
  uint64_t len64 = len >> 4;
  uint64_t xor64 = 0;
  for (i = 0; i < len64; i++) {
    xor64 = xor64 ^ data64[i];
  }
  xor = xor ^ ((uint32_t) xor64) ^ ((uint32_t) (xor64 >> 32));
  len = len & 0x7;
  data += len64 << 3;
  for (i = 0; i < len; i++) {
    xor = xor ^ (data[i] << ((i << 3) & 0x1F));
  }
  return xor;
}

static inline uint32_t initial_chk (local_state_t *local)
{
  return local->params.use_crc_check ? 0xffffffff : 0x0;
}

static uint32_t calculate_chk (local_state_t *local,
                               uint32_t len, uint8_t *data, uint32_t chk)
{
  if (local->params.use_crc_check) {
    uint32_t crc;
    crc = crc32(0, Z_NULL, 0);
    return crc32(crc, data, len);
  }
  else if (local->params.use_xor_check) {
    if (__WORDSIZE == 64) {
      return calculate_xor_64(len, data, chk);
    }
    else {
      return calculate_xor_32(len, data, chk);
    }
  }
  else {
    return chk;
  }
}

static void random_data (uint32_t *seedp, uint32_t len, uint8_t *data)
{
  uint32_t i;
  uint32_t *data32 = (uint32_t *) data;
  uint32_t len32 = len >> 2;
  for (i = 0; i < len32; i++) {
    data32[i] = (uint32_t) rand_r(seedp);
  }
  len = len & 0x3;
  data += len32 << 2;
  for (i = 0; i < len; i++) {
    data[i] = (uint8_t) myrand_r(seedp, 0, 255);
  }
}

static void init_data (uint32_t *seedp, uint32_t len, uint8_t *data,
                       int randomize, uint8_t byte)
{
  if (randomize) {
    random_data(seedp, len, data);
  }
  else {
    memset(data, byte, len);
  }
}

static void print_hdr (FILE *out, const char *string,
                       local_state_t *local, msg_hdr_t *hdr)
{
  if (!local->params.quiet) {
    fprintf(out, "%s rank %d : %s header "
                 "src=%d (%s) -> dst=%d (%s) tag=0x%lx len=%lu "
                 "seq=%lu hdr_chk=0x%08lx data_chk=0x%08lx seed=0x%08lx\n",
            myhostname,
	    (int) local->comm_rank, string,
            (int) hdr->src, get_hostname(local, hdr->src), 
            (int) hdr->dst, get_hostname(local, hdr->dst),
            (unsigned long) hdr->tag, (unsigned long) hdr->len,
	    (unsigned long) hdr->seq, (unsigned long) hdr->hdr_chk,
	    (unsigned long) hdr->data_chk, (unsigned long) hdr->seed);
  }
}

static int print_data_errors (FILE *out,
			      local_state_t *local, msg_state_t *msg,
			      uint32_t src_seed, uint32_t src_seq,
			      uint32_t len, uint8_t *data)
{
  /* Use src_seed to find and print out incorrect bytes */
  /* Use dst_seed to find and print out unwritten bytes (poison & zero) */
  int errors = 0;
  uint32_t i, j;
  uint8_t *src_data = mymalloc(len);
  uint8_t *dst_data = mymalloc(len);
  uint8_t *address;
  uint32_t dst_seed = msg->poison_seed;
  int initial = (local->params.poison || local->params.zero);
  init_data(&src_seed, len, src_data, local->params.use_random_data,
	    local->params.initial_byte);
  if (local->params.poison) {
    init_data(&dst_seed, len, dst_data, 1, 0x00);
  }
  else if (local->params.zero) {
    init_data(&dst_seed, len, dst_data, 0, 0x00);
  }
  if (!local->params.quiet) {
    fprintf(out, "%s rank %d : found following data errors\n",
            myhostname, (int) local->comm_rank);
    fprintf(out, "%s rank %d : %12s\t%12s\t%7s\t%7s",
            myhostname, (int) local->comm_rank,
            "Address", "Offset", "Expect", "Actual");
    if (initial) {
      fprintf(out, "\t%8s", "Initial");
    }
    fprintf(out, "\n");
  }
  for (i = 0; i < len; i += 16) {
    for (j = 0; j < 16 && i + j < len; j++) {
      if (src_data[i+j] != data[i+j]) {
        if (!local->params.quiet) {
	  if (errors < local->params.max_errors) {
            fprintf(out, "%s rank %d : %12p\t%12d\t   0x%02x\t   0x%02x",
                    myhostname, (int) local->comm_rank,
		    &data[i+j], i + j, src_data[i+j], data[i+j]);
	    if (initial) {
	      fprintf(out, "\t    0x%02x", dst_data[i+j]);
	    }
	    fprintf(out, "\n");
	  }
	  else if (errors == local->params.max_errors) {
            fprintf(out, "%s rank %d : too many errors so no more printing\n",
		    myhostname, (int) local->comm_rank);
	  }
	}
	errors++;
      }
    }
  }
  if (!local->params.quiet) {
    fprintf(out, "%s rank %d : %d data error(s) in %d data bytes\n",
            myhostname, (int) local->comm_rank, errors, len);
  }
  if (local->params.verbose) {
    fprintf(out, "Byte summary in 64B chunks (.=good, X=bad, I=init)\n");
    address = (uint8_t *) (((intptr_t) data) & ~0x3Full);
    while (address < &data[len]) {
      if ((((intptr_t) address) & 0x3Full) == 0) {
        fprintf(out, "\n%12p  ", address);
      }
      if (address >= data) {
        i = address - data;
        if (*address == src_data[i])
          fprintf(out, ".");
        else if (*address == dst_data[i])
          fprintf(out, "I");
        else
          fprintf(out, "X");
      }
      else {
        fprintf(out, " ");
      }
      address++;
    }
    fprintf(out, "\n");
  }
  myfree(src_data);
  myfree(dst_data);
  return errors;
}

static void generate_small_msg (local_state_t *local, small_msg_t *msg, uint32_t len,
				uint8_t *data, uint32_t imm_data)
{
  switch (local->test_mode) {
  case SMALL_ONE_BYTE:
    msg->data_one_byte = imm_data;
    break;
  case SMALL_TWO_BYTE:
    msg->data_two_byte = imm_data;
    break;
  case SMALL_MULTI_BYTE:
    msg->data_multi_byte_chk = calculate_xor_16(len - sizeof(small_msg_hdr_t), data, initial_chk(local));
    break;
  }
  return;
}

static void generate_msg (local_state_t *local, 
                          msg_hdr_t *hdr, uint32_t dst, uint32_t tag,
			  uint32_t len, uint8_t *data)
{
  uint32_t hdr_chk;
  int recalculate_data_chk = 0;
  assert(dst < local->comm_size);
  hdr->src = local->comm_rank;
  hdr->dst = dst;
  hdr->tag = tag;
  hdr->len = len;
  hdr->seq = local->peers[dst].send_seq++;
  hdr->hdr_chk = 0;
  if (local->params.use_random_length) {
    recalculate_data_chk = local->params.data_check;
  }
  if (recalculate_data_chk) {
    uint32_t data_len = len - sizeof(msg_hdr_t);
    hdr->data_chk = calculate_chk(local, data_len, data, initial_chk(local));
  }
  hdr_chk = calculate_chk(local, sizeof(msg_hdr_t), (uint8_t *) hdr, 
                          initial_chk(local));
  hdr->hdr_chk = hdr_chk;
}

static int validate_uint32 (FILE *out,
			    local_state_t *local, const char *name,
			    uint32_t value, uint32_t expected)
{
  if (value == expected) {
    return 0;
  }
  else {
    if (!local->params.quiet) {
      fprintf(out, "%s rank %d : %s error, got 0x%lx instead of 0x%lx\n",
              myhostname,
	      (int) local->comm_rank, name,
              (unsigned long) value, (unsigned long) expected);
    }
    return 1;
  }
}

static int validate_result (local_state_t *local,
                            const char *name, int result)
{
  if (result == MPI_SUCCESS) {
    return 0;
  }
  else {
    if (!local->params.quiet) {
      printf("%s rank %d : %s returned result code %d\n",
             myhostname,
	     (int) local->comm_rank, name, result);
    }
    return 1;
  }
}

static int validate_small_msg (FILE *out, local_state_t *local,
			 msg_hdr_t *hdr, uint32_t len, uint8_t *data)
{
  uint32_t data_len;
  int errors = 0;
  uint32_t chk, data_chk;
  small_msg_t *msg = (small_msg_t *) hdr;

  switch (local->test_mode) {
  case SMALL_ONE_BYTE:
    if (local->recv_validate_buf)
      local->recv_validate_buf[msg->data_one_byte & 0xff]++;
    break;
  case SMALL_TWO_BYTE:
    if (local->recv_validate_buf)
      local->recv_validate_buf[msg->data_two_byte & 0xffff]++;
    break;
  case SMALL_MULTI_BYTE:
    data = (uint8_t *) (msg + 1);
    data_len = len - sizeof(small_msg_hdr_t);
    chk = 0;
    data_chk = msg->data_multi_byte_chk;
    chk = calculate_xor_16(data_len, data, initial_chk(local));
    errors += validate_uint32(out, local, "data_chk", data_chk, chk);
    break;
  }
  return errors;
}

static int validate_msg (FILE *out, local_state_t *local, msg_state_t *msg,
			 msg_hdr_t *hdr, uint32_t src, uint32_t tag,
			 uint32_t len, uint8_t *data)
{
  uint32_t data_len = len - sizeof(msg_hdr_t);
  int errors = 0;
  uint32_t chk, hdr_chk, data_chk, hdr_ok;

  if (local->test_mode != STANDARD) {
    return validate_small_msg(out, local, hdr, len, data);
  }

  errors += validate_uint32(out, local, "src", hdr->src, src);
  errors += validate_uint32(out, local, "dst", hdr->dst, local->comm_rank);
  errors += validate_uint32(out, local, "tag", hdr->tag, tag);
  errors += validate_uint32(out, local, "len", hdr->len, len);
  errors += validate_uint32(out, local, "seq",
                            hdr->seq, local->peers[src].recv_seq);
  local->peers[src].recv_seq++;

  hdr_chk = hdr->hdr_chk;
  hdr->hdr_chk = 0;
  chk = calculate_chk(local, sizeof(msg_hdr_t), (uint8_t *) hdr,
                     initial_chk(local));
  hdr->hdr_chk = hdr_chk;
  errors += validate_uint32(out, local, "hdr_chk", hdr_chk, chk);
  hdr_ok = (chk == hdr_chk);

  chk = 0;
  data_chk = hdr->data_chk;
  if (local->params.data_check) {
    chk = calculate_chk(local, data_len, data, initial_chk(local));
  }
  errors += validate_uint32(out, local, "data_chk", data_chk, chk);
  if (errors) {
    print_hdr(out, "received", local, hdr);
    if (chk != data_chk) {
      if (hdr_ok) {
        errors += print_data_errors(out, local, msg, hdr->seed, hdr->seq,
				    data_len, data);
      }
      else if (!local->params.quiet) {
	fprintf(out, "%s rank %d : bad header, so not printing data errors\n",
                myhostname, (int) local->comm_rank);
      }
    }
  }
  return errors;
}

static void alloc_msg_array (msg_array_t *array, uint32_t count,
			     uint32_t size, uint32_t align, uint32_t hdr_size)
{
  uint32_t i;

#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
  unsigned f = 1;
#endif

  assert(size >= hdr_size);
  array->count = count;
  array->size = size;
  array->index = 0;
  array->msgs = (msg_state_t *) mymalloc(count * sizeof(msg_state_t));
  for (i = 0; i < count; i++) {
    array->msgs[i].buf = (uint8_t *) mymemalign(size, align);
#ifdef MPI_STRESS_CUDA
    CUDA_CALL(cudaMalloc, (void **) &array->msgs[i].gpubuf, size);
#endif
#ifdef MPI_STRESS_ONEAPI
    oneapi_ze_malloc((void **) &array->msgs[i].gpubuf, size);
#endif
    array->msgs[i].req = MPI_REQUEST_NULL;
    array->msgs[i].poison_seed = 0;
  }
}

static void preinit_msg (local_state_t *local, msg_state_t *msg,
                         uint32_t data_len, int src_data)
{
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
  uint8_t *gpubuf = msg->gpubuf;
  msg_hdr_t *gpuhdr = (msg_hdr_t *) gpubuf;
  uint8_t *gpudata = (uint8_t *) (gpuhdr + 1);
#endif
  uint8_t *buf = msg->buf;
  msg_hdr_t *hdr = (msg_hdr_t *) buf;
  uint8_t *data = (uint8_t *) (hdr + 1);
  memset(hdr, 0, sizeof(msg_hdr_t));
  msg->poison_seed = 0;
  if (src_data) {
    /* initialization for source data buffers (to send from) */
    hdr->seed = local->seed;
    init_data(&local->seed, data_len, data,
              local->params.use_random_data,
	      local->params.initial_byte);
    if (local->params.data_check) {
      hdr->data_chk = calculate_chk(local, data_len, data, initial_chk(local));
    }
  }
  else {
    /* initialization for destination data buffers (to receive into) */
    if (local->params.poison) {
      msg->poison_seed = local->seed;
      init_data(&local->seed, data_len, data, 1, 0x00);
    }
    else if (local->params.zero) {
      init_data(&local->seed, data_len, data, 0, 0x00);
    }
  }
#ifdef MPI_STRESS_CUDA
  CUDA_CALL(cudaMemcpy, gpubuf, buf, data_len + sizeof(msg_hdr_t), cudaMemcpyHostToDevice);
#endif
#ifdef MPI_STRESS_ONEAPI
  oneapi_ze_memcpy(gpubuf, buf, data_len + sizeof(msg_hdr_t));
#endif
}

static void preinit_small_msg (local_state_t *local, msg_state_t *msg,
                         uint32_t data_len, int src_data)
{
  uint8_t *buf = msg->buf;
  small_msg_hdr_t *hdr = (small_msg_hdr_t *) buf;
  uint8_t *data = (uint8_t *) (hdr + 1);
  memset(hdr, 0, sizeof(small_msg_hdr_t));
  if (src_data) {
    /* initialization for source data buffers (to send from) */
    init_data(&local->seed, data_len, data,
              local->params.use_random_data,
	      local->params.initial_byte);
    *hdr = calculate_xor_16(data_len, data, initial_chk(local));
  }
  else {
    /* initialization for destination data buffers (to receive into) */
    if (local->params.poison) {
      init_data(&local->seed, data_len, data, 1, 0x00);
    }
    else if (local->params.zero) {
      init_data(&local->seed, data_len, data, 0, 0x00);
    }
  }
}

static void preinit_msg_array (local_state_t *local, msg_array_t *array,
			       int src_data)
{
  uint32_t i;
  uint32_t max_data_len;
  if (local->test_mode == STANDARD) {
    max_data_len = array->size - sizeof(msg_hdr_t);
    for (i = 0; i < array->count; i++) {
      preinit_msg(local, &array->msgs[i], max_data_len, src_data);
    }
  } else {
    max_data_len = array->size - sizeof(small_msg_hdr_t);
    for (i = 0; i < array->count; i++) {
      preinit_small_msg(local, &array->msgs[i], max_data_len, src_data);
    }
  }
#ifdef MPI_STRESS_CUDA
  for (i = 0; i < array->count; i++) {
    CUDA_CALL(cudaMemcpyAsync, array->msgs[i].gpubuf, array->msgs[i].buf,
	      array->size, cudaMemcpyHostToDevice, 0);
  }
  CUDA_CALL(cudaDeviceSynchronize);
#endif
#ifdef MPI_STRESS_ONEAPI
  for (i = 0; i < array->count; i++) {
    oneapi_ze_memcpy(
      array->msgs[i].gpubuf, array->msgs[i].buf, array->size);
  }
#endif
}

static void free_msg_array (msg_array_t *array)
{
  uint32_t i;
  for (i = 0; i < array->count; i++) {
    myfree(array->msgs[i].buf);
#ifdef MPI_STRESS_CUDA
    CUDA_CALL(cudaFree, array->msgs[i].gpubuf);
#endif
#ifdef MPI_STRESS_ONEAPI
    MPI_STRESS_ONEAPI_ZE_CALL(zeMemFree, ze_context, array->msgs[i].gpubuf);
#endif
  }
  myfree(array->msgs);
}

static uint32_t choose_dst (local_state_t *local)
{
  uint32_t r;
  if (local->params.round_robin) {
    r = local->send_round_robin;
    local->send_round_robin++;
    if (local->send_round_robin >= local->send_size) {
      local->send_round_robin = 0;
    }
  }
  else {
    r = myrand_r(&local->seed, 0, local->send_size - 1);
  }
  return local->send_mapping[r];
}

static uint32_t choose_tag (local_state_t *local)
{
  uint32_t tag = myrand_r(&local->seed, local->min_tag, local->max_tag);
  assert(tag != (uint32_t) MPI_ANY_TAG);
  return tag;
}

static uint32_t choose_size (local_state_t *local)
{
  uint32_t min = local->params.min_msg_size;
  uint32_t max = local->params.max_msg_size;
  uint32_t incr = local->params.incr_msg_size;
  uint32_t size;
  if (incr > 0) {
    uint32_t range = (max - min) / incr;
    uint32_t n = myrand_r(&local->seed, 0, range);
    size = min + n * incr;
    size = MIN(size, max);
  }
  else {
    uint32_t count = 0;
    uint32_t msg_size = min;
    uint32_t n;
    while (msg_size <= max) {
      count++;
      msg_size <<= 1;
    }
    n = myrand_r(&local->seed, 0, count - 1);
    size = min << n;
  }
  assert(size >= min && size <= max);
  return size;
}

static int send_one_msg (local_state_t *local, msg_array_t *send_array,
                         int size, int *sent, int imm_data)
{
  int errors = 0;
  int result;
  uint32_t index = send_array->index;
  uint8_t *buf = send_array->msgs[index].buf;
  msg_hdr_t *hdr = (msg_hdr_t *) buf;
  uint8_t *data = (uint8_t *) (hdr + 1);
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
  uint8_t *gpubuf = send_array->msgs[index].gpubuf;
  msg_hdr_t *gpuhdr = (msg_hdr_t *) gpubuf;
  uint8_t *gpudata = (uint8_t *) (gpuhdr + 1);
#endif
  uint32_t dst;
  uint32_t tag;
  MPI_Request *send_req = &send_array->msgs[index].req;
  if (*send_req != MPI_REQUEST_NULL) {
    MPI_Status send_status;
    int flag;

    result = MPI_Test(send_req, &flag, &send_status);
    errors += validate_result(local, "MPI_Test", result);
    if (!flag) {
      *sent = 0;
      return errors;
    }
  }
  dst = choose_dst(local);
  if (local->test_mode == STANDARD){
    tag = choose_tag(local);
    generate_msg(local, hdr, dst, tag, size, data);
#ifdef MPI_STRESS_CUDA
    if (local->params.use_gpu_send)
      CUDA_CALL(cudaMemcpy, gpubuf, buf, sizeof(msg_hdr_t), cudaMemcpyHostToDevice);
#endif
#ifdef MPI_STRESS_ONEAPI
    if (local->params.use_gpu_send)
      oneapi_ze_memcpy(gpubuf, buf, sizeof(msg_hdr_t));
#endif
  } else {
    tag = 1;
    data = (uint8_t *) (((small_msg_hdr_t*) send_array->msgs[index].buf) + 1);
    generate_small_msg(local, (small_msg_t *) hdr, size, data, imm_data);
#ifdef MPI_STRESS_CUDA
    if (local->params.use_gpu_send)
      CUDA_CALL(cudaMemcpy, gpubuf, buf, size, cudaMemcpyHostToDevice);
#endif
#ifdef MPI_STRESS_ONEAPI
    if (local->params.use_gpu_send)
      oneapi_ze_memcpy(gpubuf, buf, size);
#endif
  }

#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
  if (local->params.use_gpu_send)
    result = local->params.mpi_send_fn(gpuhdr, size, MPI_BYTE, dst, tag,
				       MPI_COMM_WORLD, send_req);
  else
#endif
  result = local->params.mpi_send_fn(hdr, size, MPI_BYTE, dst, tag,
				     MPI_COMM_WORLD, send_req);
  errors += validate_result(local, local->params.mpi_send_fn_name, result);
  if (local->params.verbose >= 2) {
    print_hdr(stdout, "send", local, hdr);
  }
  if (local->params.progress) {
    printf("S");
  }
  send_array->index++;
  if (send_array->index == send_array->count) {
    send_array->index = 0;
  }
  *sent = 1;
  return errors;
}

static int poll_for_msgs (local_state_t *local, msg_array_t *recv_array,
                          int size)
{
  FILE *out = stdout;
  int errors = 0;
  int result, flag;
  int progress = 1;
  int irecv_size = (size == RANDOM_SIZE) ? local->params.max_msg_size : size;
  while (progress) {
    uint32_t index = recv_array->index;
    msg_state_t *msg = &recv_array->msgs[index];
    uint8_t *buf = msg->buf;
    msg_hdr_t *hdr = (msg_hdr_t *) buf;
    uint8_t *data = (uint8_t *) (hdr + 1);
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
    uint8_t *gpubuf = msg->gpubuf;
    msg_hdr_t *gpuhdr = (msg_hdr_t *) gpubuf;
    uint8_t *gpudata = (uint8_t *) (gpuhdr + 1);
#endif
    MPI_Request *recv_req = &msg->req;
    MPI_Status recv_status;
    uint32_t max_data_len = recv_array->size - sizeof(msg_hdr_t);

    progress = 0;
    if (*recv_req != MPI_REQUEST_NULL) {
      result = MPI_Test(recv_req, &flag, &recv_status);
      errors += validate_result(local, "MPI_Test", result);
      if (flag) {
        int src = recv_status.MPI_SOURCE;
        int tag = recv_status.MPI_TAG;
        int count;
        result = MPI_Get_count(&recv_status, MPI_BYTE, &count);
        errors += validate_result(local, "MPI_Get_count", result);
        if (count == 0) {
          /* got a final message */
          errors += validate_uint32(out, local, "source",
                                    local->peers[src].completed, 0);
          if (local->peers[src].completed == 0) {
            local->peers[src].completed = 1;
            local->completed++;
          }
          if (local->params.verbose) {
            printf("Received final message from rank %d to %d\n",
                   src, (int) local->comm_rank);
          }
          if (local->params.progress) {
             printf("r");
          }
        }
        else {
	  if (size != RANDOM_SIZE) {
            errors += validate_uint32(out, local, "length", count, size);
	  }
#ifdef MPI_STRESS_CUDA
	  if (local->params.use_gpu_recv)
	    CUDA_CALL(cudaMemcpy, buf, gpubuf, irecv_size, cudaMemcpyDeviceToHost);
#endif
#ifdef MPI_STRESS_ONEAPI
	  if (local->params.use_gpu_recv)
      oneapi_ze_memcpy(buf, gpubuf, irecv_size);
#endif
          errors += validate_msg(stdout, local, msg, hdr, src, tag,
				 count, data);
          if (local->params.verbose >= 2) {
            print_hdr(stdout, "recv", local, hdr);
          }
          if (local->params.progress) {
            printf("R");
          }
	  if (local->test_mode == STANDARD) {
	    preinit_msg(local, msg, max_data_len, 0);
	  }
        }
	progress = 1;
      }
    }
    if (*recv_req == MPI_REQUEST_NULL) {
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
      if (local->params.use_gpu_recv)
        result = MPI_Irecv(gpuhdr, irecv_size, MPI_BYTE, MPI_ANY_SOURCE,
                           MPI_ANY_TAG, MPI_COMM_WORLD, recv_req);
      else
#endif
      result = MPI_Irecv(hdr, irecv_size, MPI_BYTE, MPI_ANY_SOURCE,
                         MPI_ANY_TAG, MPI_COMM_WORLD, recv_req);
      errors += validate_result(local, "MPI_Irecv", result);
      recv_array->index++;
      if (recv_array->index == recv_array->count) {
        recv_array->index = 0;
      }
      progress = 1;
    }
  }
  return errors;
}

static int wait_for_sends (local_state_t *local, msg_array_t *send_array,
                           msg_array_t *recv_array, uint32_t size)
{
  int errors = 0;
  int result;
  uint32_t i;
  for (i = 0; i < send_array->count; i++) {
    while (send_array->msgs[i].req != MPI_REQUEST_NULL) {
      MPI_Status send_status;
      int flag;
      result = MPI_Test(&send_array->msgs[i].req, &flag, &send_status);
      errors += validate_result(local, "MPI_Test", result);
      errors += poll_for_msgs(local, recv_array, size); 
    }
  }
  return errors;
}

static int final_sends (local_state_t *local, msg_array_t *recv_array,
                        uint32_t size)
{
  /* The final sends are distinguished by having zero length (no msg hdr) */
  int errors = 0;
  int result;
  uint32_t i;
  for (i = 0; i < local->comm_size; i++) {
    if (i != local->comm_rank) {
      result = MPI_Send(NULL, 0, MPI_BYTE, i, 0, MPI_COMM_WORLD);
      errors += validate_result(local, "MPI_Send", result);
      if (local->params.verbose) {
        printf("Sent final message from rank %d to %d\n",
               (int) local->comm_rank, i);
      }
      if (local->params.progress) {
        printf("s");
      }
      errors += poll_for_msgs(local, recv_array, size); 
    }
  }
  return errors;
}

static int wait_for_recvs (local_state_t *local, msg_array_t *recv_array,
                           uint32_t size)
{
  int errors = 0;
  while (local->completed != local->comm_size - 1) {
    errors += poll_for_msgs(local, recv_array, size);
  }
  return errors;
}

static int cancel_recvs (local_state_t *local, msg_array_t *recv_array)
{
  int errors = 0;
  int result;
  uint32_t i;
  for (i = 0; i < recv_array->count; i++) {
    if (recv_array->msgs[i].req != MPI_REQUEST_NULL) {
      result = MPI_Cancel(&recv_array->msgs[i].req);
      errors += validate_result(local, "MPI_Cancel", result);
      result = MPI_Wait(&recv_array->msgs[i].req, MPI_STATUS_IGNORE);
      errors += validate_result(local, "MPI_Wait", result);
    }
  }
  return errors;
}

static int run_random_size (local_state_t *local, uint32_t msgs,
			    uint32_t max_size)
{
  int errors = 0;
  uint32_t i;
  double start, finish, secs, min_secs, max_secs, sum_secs, bw, rate;
  msg_array_t send_array, recv_array;
  int result;
  double total_size = 0.0;
  double sum_total_size = 0.0;

  alloc_msg_array(&send_array, local->params.window_size,
                  max_size, local->params.align, 0);
  preinit_msg_array(local, &send_array, 1);
  alloc_msg_array(&recv_array, local->params.window_size,
                  max_size, local->params.align, 0);
  preinit_msg_array(local, &recv_array, 0);

  local->completed = 0;
  for (i = 0; i < local->comm_size; i++) {
    local->peers[i].completed = (i == local->comm_size);
  }

  result = MPI_Barrier(MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Barrier", result);
  start = MPI_Wtime();
  errors += poll_for_msgs(local, &recv_array, RANDOM_SIZE);
  if (local->send_size > 0) {
    uint32_t m = 0;
    while (m < msgs) {
      uint32_t size = choose_size(local);
      int sent;
      errors += send_one_msg(local, &send_array, size, &sent, 0);
      errors += poll_for_msgs(local, &recv_array, RANDOM_SIZE);
      if (sent) {
	m++;
        total_size += (double) size;
      }
    }
    errors += wait_for_sends(local, &send_array, &recv_array, RANDOM_SIZE);
  }
  errors += final_sends(local, &recv_array, RANDOM_SIZE);
  errors += wait_for_recvs(local, &recv_array, RANDOM_SIZE);
  finish = MPI_Wtime();
  errors += cancel_recvs(local, &recv_array);
  if (local->params.progress) {
    printf("\n");
  }
  secs = finish - start;
  result = MPI_Allreduce(&secs, &min_secs, 1,
                         MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Allreduce", result);
  result = MPI_Allreduce(&secs, &max_secs, 1,
                         MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Allreduce", result);
  result = MPI_Allreduce(&secs, &sum_secs, 1,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Allreduce", result);
  result = MPI_Allreduce(&total_size, &sum_total_size, 1,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Allreduce", result);
  secs = min_secs;	/* min_secs or max_secs or sum_secs/local->comm_size */
  bw = (secs == 0.0) ? 0.0 : (sum_total_size / local->comm_size) / secs;
  rate = (secs == 0.0) ? 0.0 : ((double) msgs) / secs;

  if (local->comm_rank == 0) {
    printf("%15s\t%15lu\t%18.2f\t%15.2f\n",
           "Randomized", (unsigned long) msgs, bw / 1000000.0, rate);
  }

  free_msg_array(&send_array);
  free_msg_array(&recv_array);

  return errors;
}

static int run_random_sizes (local_state_t *local)
{
  /* the random run is parameterized by the maximum message size but
   * each message size will be randomized between the min and max. */
  uint32_t msg_size = local->params.max_msg_size;
  uint32_t msgs = (msg_size == 0) ? 0 : local->params.max_data / msg_size;
  msgs = MAX(msgs, local->params.min_msgs);
  msgs = MIN(msgs, local->params.max_msgs);

  if (local->comm_rank == 0) {
    printf("\n%15s\t%15s\t%18s\t%15s\n",
           "Size (bytes)", "TxMessages",
           "TxMillionBytes/s", "TxMessages/s");
  }

  return run_random_size(local, msgs, msg_size);
}

int check_recv_validate_buf(local_state_t *local, uint32_t msgs){
  int i, errors = 0;

  switch(local->test_mode){
  case SMALL_ONE_BYTE:
    for (i = 0; i < 256; i++) {
      if (local->recv_validate_buf) {
	uint8_t expected = (msgs / 256) + (i < msgs % 256);
	if (local->recv_validate_buf[i] != expected) {
	  printf("smallmsg error: recv_buf[%d] = %d, expected %d\n", i, local->recv_validate_buf[i], expected);
	  errors++;
	}
      }
    }
    break;
  case SMALL_TWO_BYTE:
    for (i = 0; i < msgs; i++) {
      if (local->recv_validate_buf) {
	if ((uint16_t)local->recv_validate_buf[i] != 1) {
	  printf("smallmsg error: recv_buf[%d] = %d, expected %d\n", i, local->recv_validate_buf[i], 1);
	  errors++;
	}
      }
    }
    break;
  }

  return errors;
}

static int run_one_size(local_state_t *local, uint32_t msgs, uint32_t size)
{
  int errors = 0;
  uint32_t i;
  double start, finish, secs, min_secs, max_secs, sum_secs, bw, rate;
  msg_array_t send_array, recv_array;
  int result;
  uint32_t hdr_size;

  switch (local->test_mode) {
  case SMALL_ONE_BYTE:
  case SMALL_TWO_BYTE:
    hdr_size = 0;
    break;
  case SMALL_MULTI_BYTE:
    hdr_size = sizeof(small_msg_hdr_t);
    break;
  case STANDARD:
  default:
    hdr_size = sizeof(msg_hdr_t);
    break;
  }

  alloc_msg_array(&send_array, local->params.window_size,
		  size, local->params.align, hdr_size);
  alloc_msg_array(&recv_array, local->params.window_size,
		  size, local->params.align, hdr_size);

  switch (local->test_mode) {
  case SMALL_ONE_BYTE:
  case SMALL_TWO_BYTE:
    local->recv_validate_buf = mycalloc(msgs, sizeof(uint16_t));
    break;
  case SMALL_MULTI_BYTE:
  case STANDARD:
  default:
    preinit_msg_array(local, &send_array, 1);
    preinit_msg_array(local, &recv_array, 0);
    break;
  }

  local->completed = 0;
  for (i = 0; i < local->comm_size; i++) {
    local->peers[i].completed = (i == local->comm_size);
  }

  result = MPI_Barrier(MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Barrier", result);
  start = MPI_Wtime();
  errors += poll_for_msgs(local, &recv_array, size);
  if (local->send_size > 0) {
    uint32_t m = 0;
    while (m < msgs) {
      int sent;
      errors += send_one_msg(local, &send_array, size, &sent, m);
      errors += poll_for_msgs(local, &recv_array, size);
      m += sent;
    }
    errors += wait_for_sends(local, &send_array, &recv_array, size);
  }
  errors += final_sends(local, &recv_array, size);
  errors += wait_for_recvs(local, &recv_array, size);
  finish = MPI_Wtime();
  errors += cancel_recvs(local, &recv_array);
  if (local->params.progress) {
    printf("\n");
  }
  secs = finish - start;
  result = MPI_Allreduce(&secs, &min_secs, 1,
                         MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Allreduce", result);
  result = MPI_Allreduce(&secs, &max_secs, 1,
                         MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Allreduce", result);
  result = MPI_Allreduce(&secs, &sum_secs, 1,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  errors += validate_result(local, "MPI_Allreduce", result);

  if ((local->test_mode == SMALL_ONE_BYTE) || (local->test_mode == SMALL_TWO_BYTE))
    errors += check_recv_validate_buf(local, msgs);

  secs = min_secs;	/* min_secs or max_secs or sum_secs/local->comm_size */
  bw = (secs == 0.0) ? 0.0 : ((double) msgs * (double) size) / secs;
  rate = (secs == 0.0) ? 0.0 : ((double) msgs) / secs;

  if (local->comm_rank == 0) {
    printf("%15lu\t%15lu\t%18.2f\t%15.2f\n", 
           (unsigned long) size, (unsigned long) msgs, bw / 1000000.0, rate);
  }

  free_msg_array(&send_array);
  free_msg_array(&recv_array);
  if(local->recv_validate_buf) {
    myfree(local->recv_validate_buf);
    local->recv_validate_buf = NULL;
  }
  return errors;
}

static int run_all_sizes (local_state_t *local)
{
  uint32_t msg_size = local->params.min_msg_size;
  int errors = 0;

  if (local->comm_rank == 0) {
    printf("\n%15s\t%15s\t%18s\t%15s\n",
           "Size (bytes)", "TxMessages",
           "TxMillionBytes/s", "TxMessages/s");
  }

  local->test_mode = STANDARD;
  while (msg_size <= local->params.max_msg_size) {
    uint32_t msgs = (msg_size == 0) ? 0 : local->params.max_data / msg_size;
    msgs = MAX(msgs, local->params.min_msgs);
    msgs = MIN(msgs, local->params.max_msgs);

    if (local->params.verbose && local->comm_rank == 0) {
      printf("Message size %10lu : %lu messages, %lu bytes\n",
	     (unsigned long) msg_size, (unsigned long) msgs, 
	     (unsigned long) (msg_size * msgs));
    }

    errors += run_one_size(local, msgs, msg_size);

    if (local->params.incr_msg_size) {
      msg_size += local->params.incr_msg_size;
    }
    else {
      msg_size = (msg_size == 0) ? 1 : (msg_size << 1);
    }
  }
  return errors;
}

static int run_one_misaligned_size (local_state_t *local, uint32_t msgs, uint32_t size)
{
  int errors = 0;
  uint32_t i;
  double start, finish, secs, min_secs, max_secs, sum_secs, bw, rate;
  msg_array_t send_array, recv_array;
  int result;
  int offset_tx, offset_rx;

  local->params.align = 8;

  alloc_msg_array(&send_array, local->params.window_size,
		  size+8, local->params.align, 0);
  alloc_msg_array(&recv_array, local->params.window_size,
		  size+8, local->params.align, 0);

  send_array.size -= 8;
  recv_array.size -= 8;

  for (offset_tx = 0; offset_tx < 8; offset_tx++) {
    for (offset_rx = 0; offset_rx < 8; offset_rx++) {
      for (i = 0; i < send_array.count; i++) {
	send_array.msgs[i].buf += offset_tx;
	send_array.msgs[i].req = MPI_REQUEST_NULL;
	send_array.msgs[i].poison_seed = 0;
	send_array.index = 0;
      }
      for (i = 0; i < recv_array.count; i++) {
	recv_array.msgs[i].buf += offset_rx;
	recv_array.msgs[i].req = MPI_REQUEST_NULL;
	recv_array.msgs[i].poison_seed = 0;
	recv_array.index = 0;
      }
      /* GPUs do not allow misalligned buffers */

      preinit_msg_array(local, &send_array, 1);
      preinit_msg_array(local, &recv_array, 0);

      local->completed = 0;
      for (i = 0; i < local->comm_size; i++) {
	local->peers[i].completed = (i == local->comm_size);
      }

      result = MPI_Barrier(MPI_COMM_WORLD);
      errors += validate_result(local, "MPI_Barrier", result);
      start = MPI_Wtime();
      errors += poll_for_msgs(local, &recv_array, size);
      if (local->send_size > 0) {
	uint32_t m = 0;
	while (m < msgs) {
	  int sent;
	  errors += send_one_msg(local, &send_array, size, &sent, 0);
	  errors += poll_for_msgs(local, &recv_array, size);
	  m += sent;
	}
	errors += wait_for_sends(local, &send_array, &recv_array, size);
      }
      errors += final_sends(local, &recv_array, size);
      errors += wait_for_recvs(local, &recv_array, size);
      finish = MPI_Wtime();
      errors += cancel_recvs(local, &recv_array);
      if (local->params.progress) {
	printf("\n");
      }
      secs = finish - start;
      result = MPI_Allreduce(&secs, &min_secs, 1,
			     MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      errors += validate_result(local, "MPI_Allreduce", result);
      result = MPI_Allreduce(&secs, &max_secs, 1,
			     MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      errors += validate_result(local, "MPI_Allreduce", result);
      result = MPI_Allreduce(&secs, &sum_secs, 1,
			     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      errors += validate_result(local, "MPI_Allreduce", result);
      secs = min_secs;	/* min_secs or max_secs or sum_secs/local->comm_size */
      bw = (secs == 0.0) ? 0.0 : ((double) msgs * (double) size) / secs;
      rate = (secs == 0.0) ? 0.0 : ((double) msgs) / secs;

      for (i = 0; i < local->params.window_size; i++) {
	send_array.msgs[i].buf -= offset_tx;
	recv_array.msgs[i].buf -= offset_rx;
      }
      if (local->comm_rank == 0) {
	printf("%15lu\t        %1d,%1d\t%15lu\t%18.2f\t%15.2f\n",
	       (unsigned long) size, offset_tx, offset_rx, (unsigned long) msgs, bw / 1000000.0, rate);
      }
    }
  }

  send_array.size += 8;
  recv_array.size += 8;

  free_msg_array(&send_array);
  free_msg_array(&recv_array);

  if(local->recv_validate_buf) {
    myfree(local->recv_validate_buf);
    local->recv_validate_buf = NULL;
  }

  return errors;
}

static int run_misaligned_sizes (local_state_t *local)
{
  uint32_t msg_size = local->params.min_msg_size;
  int errors = 0;

  if (local->comm_rank == 0) {
    printf("\n%15s\t%13s\t%15s\t%18s\t%15s\n",
           "Size (bytes)", "Align (T,R) ", "TxMessages",
           "TxMillionBytes/s", "TxMessages/s");
  }

  while (msg_size <= local->params.max_msg_size) {
    uint32_t msgs = (msg_size == 0) ? 0 : local->params.max_data / msg_size / 64;
    msgs = MAX(msgs, local->params.min_msgs);
    msgs = MIN(msgs, local->params.max_msgs);

    if (local->params.verbose && local->comm_rank == 0) {
      printf("Message size %10lu : %lu messages, %lu bytes\n",
	     (unsigned long) msg_size, (unsigned long) msgs,
	     (unsigned long) (msg_size * msgs));
    }

    errors += run_one_misaligned_size(local, msgs, msg_size);

    if (local->params.incr_msg_size) {
      msg_size += local->params.incr_msg_size;
    } else {
      msg_size = (msg_size == 0) ? 1 : (msg_size << 1);
    }
  }
  return errors;
}

static int run_small_sizes (local_state_t *local)
{
  uint32_t msg_size;
  uint32_t msgs;
  int errors = 0;

  if (local->comm_rank == 0) {
    printf("\n%15s\t%15s\t%18s\t%15s\n",
           "Size (bytes)", "TxMessages",
           "TxMillionBytes/s", "TxMessages/s");
  }

  /* test 1B and 2B */
  msg_size = 1;
  local->test_mode = SMALL_ONE_BYTE;
  while (msg_size <= 2) {
    msgs = (msg_size == 0) ? 0 : local->params.max_data / msg_size;
    msgs = MAX(msgs, local->params.min_msgs);
    msgs = MIN(msgs, local->params.max_msgs);

    errors += run_one_size(local, msgs, msg_size);
    msg_size++;
    local->test_mode++;
  }

  /* test 3-32B */
  while (msg_size <= 32) {
    uint32_t msgs = (msg_size == 0) ? 0 : local->params.max_data / msg_size;
    msgs = MAX(msgs, local->params.min_msgs);
    msgs = MIN(msgs, local->params.max_msgs);

    if (local->params.verbose && local->comm_rank == 0) {
      printf("Message size %10lu : %lu messages, %lu bytes\n",
	     (unsigned long) msg_size, (unsigned long) msgs,
	     (unsigned long) (msg_size * msgs));
    }

    errors += run_one_size(local, msgs, msg_size);
    msg_size++;
  }
  return errors;
}

static void init_hostnames (local_state_t *local)
{
  int result;
  local->hostnames = (char *) mymalloc(MPI_MAX_PROCESSOR_NAME *
                                       local->comm_size);
  result = MPI_Allgather(myhostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         local->hostnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                         MPI_COMM_WORLD);
  validate_result(local, "MPI_Allgather", result);
}

static void init_send_mapping (local_state_t *local)
{
  uint32_t i, j;
  int *shm_comm_mapping;
  int shm_comm_me;
  uint32_t hostname_hash;
  MPI_Comm shm_comm;
  int shm_comm_rank, shm_comm_size, result;
  int *recv_size_in, *recv_size_out;
  uint32_t crc;

  local->send_size = 0;
  local->send_mapping = (uint32_t *) mymalloc(local->comm_size *
                                             sizeof(uint32_t));
  local->send_round_robin = 0;
  shm_comm_mapping = (int *) mymalloc(local->comm_size *
                                      sizeof(int));

  crc = crc32(0, Z_NULL, 0);
  hostname_hash = crc32(crc, (uint8_t *) myhostname, strlen(myhostname));
  hostname_hash &= 0x7fffffff;	/* make positive */
  result = MPI_Comm_split(MPI_COMM_WORLD, hostname_hash, local->comm_rank,
                          &shm_comm);
  validate_result(local, "MPI_Comm_split", result);
  MPI_Comm_rank(shm_comm, &shm_comm_rank);
  MPI_Comm_size(shm_comm, &shm_comm_size);
  shm_comm_me = local->comm_rank;
  result = MPI_Allgather(&shm_comm_me, 1, MPI_INT,
                         shm_comm_mapping, 1, MPI_INT, shm_comm);
  validate_result(local, "MPI_Allgather", result);
  if (local->params.verbose) {
    for (i = 0; i < (uint32_t) shm_comm_size; i++) {
      printf("Rank %d is local to rank %d on host %s\n",
             local->comm_rank, shm_comm_mapping[i], myhostname);
    }
  }
  memset(local->send_mapping, 0xff, local->comm_size * sizeof(uint32_t));
  if (local->params.grid_ndims > 0) {
    /* use grid connectivity */
    int ndims = local->params.grid_ndims;
    int *dims = (int *) mycalloc(ndims, sizeof(int));
    int *periods = (int *) mycalloc(ndims, sizeof(int));
    MPI_Comm comm_cart;
    local->grid_dims = dims;
    result = MPI_Dims_create(local->comm_size, ndims, dims);
    validate_result(local, "MPI_Dims_create", result);
    if (local->comm_rank == 0 && local->params.verbose) {
      printf("Using a %speriodic %d-dimensional grid of %d ranks with extent ",
	     (local->params.periodic_grid) ? "" : "non-",
             ndims, local->comm_size);
      for (i = 0; i < (uint32_t) ndims; i++) {
	printf("%s%d", (i == 0) ? "" : " x ", dims[i]);
      }
      printf("\n");
    }
    if (local->params.periodic_grid) {
      for (i = 0; i < (uint32_t) ndims; i++) {
	periods[i] = 1;
      }
    }
    result = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1,
                             &comm_cart);
    validate_result(local, "MPI_Cart_create", result);
    for (i = 0; i < (uint32_t) ndims; i++) {
      int rank_dec, rank_inc;
      result = MPI_Cart_shift(comm_cart, i, 1, &rank_dec, &rank_inc);
      validate_result(local, "MPI_Cart_shift", result);
      if (rank_inc != MPI_PROC_NULL) {
        assert(local->send_size < local->comm_size);
        local->send_mapping[local->send_size] = rank_inc;
        local->send_size++;
      }
      if (!local->params.uni_grid && rank_dec != MPI_PROC_NULL) {
        assert(local->send_size < local->comm_size);
        local->send_mapping[local->send_size] = rank_dec;
        local->send_size++;
      }
    }
    result = MPI_Comm_free(&comm_cart);
    validate_result(local, "MPI_Comm_free", result);
    myfree(periods);
  }
  else {
    /* use all-to-all connectivity */
    for (i = 0; i < local->comm_size; i++) {
      int map_i = 1;
      if (!local->params.include_self && i == local->comm_rank) {
        map_i = 0;
      }
      if (!local->params.include_shm && i != local->comm_rank) {
        for (j = 0; j < (uint32_t) shm_comm_size; j++) {
          if (i == (uint32_t) shm_comm_mapping[j]) {
            map_i = 0;
  	    break;
  	  }
        }
      }
      if (map_i) {
        local->send_mapping[local->send_size] = i;
        local->send_size++;
        /* Stagger the initial round-robin values */
        if (i < local->comm_rank) {
          local->send_round_robin++;
        }
      }
    }
  }
  local->send_round_robin = (local->send_size == 0) ?
                             0 : local->send_round_robin % local->send_size;
  result = MPI_Comm_free(&shm_comm);
  validate_result(local, "MPI_Comm_free", result);
  if (local->params.verbose) {
    printf("Rank %d on host %s sends to %d ranks (start round robin is %d):",
	   (int) local->comm_rank, myhostname, 
           (int) local->send_size, (int) local->send_round_robin);
    for (i = 0; i < local->send_size; i++) {
      printf(" %d", local->send_mapping[i]);
    }
    printf("\n");
  }
  recv_size_in = (int *) mycalloc(local->comm_size, sizeof(int));
  recv_size_out = (int *) mycalloc(local->comm_size, sizeof(int));
  for (i = 0; i < local->send_size; i++) {
    recv_size_in[local->send_mapping[i]]++;
  }
  result = MPI_Allreduce(recv_size_in, recv_size_out, local->comm_size,
		         MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  validate_result(local, "MPI_Allreduce", result);
  local->recv_size = recv_size_out[local->comm_rank];
  if (local->params.verbose) {
    printf("Rank %d on host %s receives from %d ranks\n",
	   (int) local->comm_rank, myhostname, 
           (int) local->recv_size);
  }
  if (local->comm_rank == 0) {
    int recv_total = 0;
    for (i = 0; i < local->comm_size; i++) {
      recv_total += recv_size_out[i];
    }
    if (recv_total == 0) {
      fatal("this configuration contains no message-passing! Please check\n"
"the number of ranks, the number of nodes, and the settings for the -i \n"
"(shared memory paths) and -s (path to self) options. Thank you.\n");
    }
  }
  myfree(recv_size_in);
  myfree(recv_size_out);
  myfree(shm_comm_mapping);
}


int run (params_t params, int comm_rank, int comm_size, uint32_t seed)
{
  local_state_t local;
  uint32_t i;
  int flag, result;
  int *mpi_tag_ub;
  int local_errors, global_errors;

  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &mpi_tag_ub, &flag);
  assert(flag);

  memset(&local, 0, sizeof(local));
  local.params = params;
  local.seed = seed;
  local.peers = mycalloc(comm_size, sizeof(peer_state_t));
  local.comm_rank = comm_rank;
  local.comm_size = comm_size;
  local.min_tag = 0;
  local.max_tag = *mpi_tag_ub;
  assert(local.max_tag > local.min_tag);
  local.completed = 0;
  init_hostnames(&local);
  init_send_mapping(&local);

  if (local.params.use_random_length)
    local_errors = run_random_sizes(&local);
  else if (local.params.use_misaligned_data)
    local_errors = run_misaligned_sizes(&local);
  else if (local.params.use_small_messages)
    local_errors = run_small_sizes(&local);
  else
    local_errors = run_all_sizes(&local);

  global_errors = 0;
  result = MPI_Allreduce(&local_errors, &global_errors, 1,
                         MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  validate_result(&local, "MPI_Allreduce", result);

  fflush(stdout);
  if (global_errors) {  
    for (i = 0; i < local.comm_size; i++) {
      result = MPI_Barrier(MPI_COMM_WORLD);
      validate_result(&local, "MPI_Barrier", result);
      if (i == (uint32_t) local.comm_rank && local_errors) {
        printf("Rank %d: %s, local errors = %d\n", 
               local.comm_rank, (local_errors ? "FAILED" : "PASSED"),
               local_errors);
        fflush(stdout);
      }
    }
  }

  myfree(local.peers);
  myfree(local.send_mapping);
  myfree(local.hostnames);
  myfree(local.grid_dims);

  return global_errors;
}

static int DEFAULT_MIN_MSG_SIZE;
static const int DEFAULT_MAX_MSG_SIZE = 4 * 1024 * 1024;
static const int DEFAULT_MIN_MSGS = 100;
static const int DEFAULT_MAX_MSGS = 10000;
static const int DEFAULT_MAX_DATA = 1024 * 1024 * 1024;

#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
static const int DEFAULT_WINDOW_SIZE = 4;
#else
static const int DEFAULT_WINDOW_SIZE = 20;
#endif
static const int DEFAULT_REPS = 1;
static const int MAX_ERRORS = 500;

static void get_help (int argc, char **argv)
{
  printf("Usage: %s [options]\n", argv[0]);
  printf("  -a INT : desired alignment for buffers. Must be power of 2 and multiple of %lu\n",
           sizeof(void *));
  printf("  -A : test misaligned buffers by adjusting start addresses\n");
  printf("  -b BYTE : byte value to initialize non-random send buffers "
	 "(otherwise 0)\n");
  printf("  -c : enable CRC checksums\n");
  printf("  -D INT : set max data amount per msg size (default %d)\n",
         DEFAULT_MAX_DATA);
  printf("  -d : enable data checksums (otherwise headers only)\n");
  printf("  -e : exercise the interconnect with random length messages\n");
  printf("  -E INT : set max number of data byte errors to show\n");
  printf("  -g INT : use INT-dimensional grid connectivity (non-periodic)\n");
  printf("  -G INT : use INT-dimensional grid connectivity (periodic)\n");
  printf("           (default is to use all-to-all connectivity)\n");
  printf("  -h : display this help page\n");
  printf("  -i : include local ranks as destinations (only for all-to-all)\n");
  printf("  -I INT : set msg size increment (default power of 2)\n");
  printf("  -l INT : set min msg size  (default %d)\n", DEFAULT_MIN_MSG_SIZE);
  printf("  -L INT : set min msg count (default %d)\n", DEFAULT_MIN_MSGS);
  printf("  -m INT : set max msg size  (default %d)\n", DEFAULT_MAX_MSG_SIZE);
  printf("  -M INT : set max msg count (default %d)\n", DEFAULT_MAX_MSGS);
  printf("  -n INT : number of times to repeat (default %d)\n", DEFAULT_REPS);
  printf("  -O : show options and parameters used for the run.\n");
  printf("  -p : show progress\n");
  printf("  -P : poison receive buffers at init and after each receive\n");
  printf("  -q : quiet mode (don't show error details)\n");
  printf("  -r : fill send buffers with random data (else 0 or -b byte)\n");
  printf("  -R : round robin destinations (default is random selection)\n");
  printf("  -s : include self as a destination (only for all-to-all)\n");
  printf("  -S : use non-blocking synchronous sends (MPI_Issend)\n");
  printf("  -t INT : run for INT minutes (implictly adds -n BIGNUM)\n");
  printf("  -u : uni-directional traffic (only for grid)\n");
  printf("  -v : enable verbose mode (more -v for more verbose)\n");
  printf("  -w INT : number of send/recv in window (default %d)\n",
         DEFAULT_WINDOW_SIZE);
  printf("  -W INT : wait on start and exit for specified seconds\n");
  printf("  -x : enable XOR checksums\n");
  printf("  -y : test small messages\n");
  printf("  -z : enable typical options for data integrity (-drx)\n");
  printf("     : (for stronger integrity checking try using -drc instead)\n");
  printf("  -Z : zero receive buffers at init and after each receive\n");
  printf("This an an MPI stress test program designed to load up an MPI\n"
         "interconnect with point-to-point messages while optionally\n"
         "checking for data integrity. By default, it runs with all-to-all\n"
         "traffic patterns, optionally including oneself and one's local\n"
         "shm peers. It can also be set up with multi-dimensional grid\n"
         "traffic patterns, and this can be parameterized to run rings,\n"
         "open 2D grids, closed 2D grids, cubic lattices, hypercubes, etc.\n"
         "Optionally, the message data can be randomized and checked\n"
         "using CRC checksums (strong but slow) or XOR checksums (weak but\n"
         "fast). The communication kernel is built out of non-blocking\n"
         "point-to-point calls to load up the interconnect. The program is\n"
         "not designed to exhaustively test out different MPI primitives.\n"
         "Performance metrics are displayed, but should be carefully\n"
         "interpreted in terms of the features enabled.\n");
}

static params_t get_params (int argc, char **argv,
                            int comm_rank, int comm_size)
{
  params_t params;
  int option;
  int result = EXIT_SUCCESS;
  int run = 1;
  const char *optlist = "a:Ab:cdD:eE:g:G:hiI:l:L:m:M:n:OpPqrRsSt:uvw:W:xyzZ";

  /* set default parameters */
  params.verbose = 0;					/* -v */
  params.progress = 0;					/* -p */
  params.show_options = 0;				/* -O */
  params.poison = 0;					/* -P */
  params.zero = 0;					/* -Z */
  params.quiet = 0;					/* -q */
  params.include_self = 0;     				/* -s */
  params.include_shm = 0;				/* -i */
  params.use_crc_check = 0;				/* -c */
  params.use_xor_check = 0;				/* -x */
  params.data_check = 0;				/* -d */
  params.use_random_data = 0;				/* -r */
  params.round_robin = 0;				/* -R */
  params.grid_ndims = 0;				/* -g %d or -G %d */
  params.periodic_grid = 0;				/* -g %d or -G %d */
  params.uni_grid = 0;					/* -u */
  params.use_random_length = 0;				/* -e */
  params.use_misaligned_data = 0;			/* -A */
  params.repeats = DEFAULT_REPS;	       		/* -n %u */
  params.runtime_mins = 0;				/* -t %u */
  params.min_msg_size = DEFAULT_MIN_MSG_SIZE;		/* -l %u */
  params.max_msg_size = DEFAULT_MAX_MSG_SIZE;		/* -m %u */
  params.incr_msg_size = 0;				/* -I %u */
  params.min_msgs = DEFAULT_MIN_MSGS;			/* -L %u */
  params.max_msgs = DEFAULT_MAX_MSGS;			/* -M %u */
  params.max_data = DEFAULT_MAX_DATA;			/* -D %u */
  params.window_size = DEFAULT_WINDOW_SIZE;		/* -w %u */
  params.align = 0;					/* -a %u */
  params.initial_byte = DEFAULT_INITIAL_BYTE;		/* -b %u */
  params.mpi_send_fn = MPI_Isend;			/* -S */
  params.mpi_send_fn_name = "MPI_Isend";		/* -S */
  params.max_errors = MAX_ERRORS;			/* -E %d */
  params.wait_on_start = 0;				/* -W %d */
  params.wait_on_exit = 0;				/* -W %d */
  params.use_small_messages = 0;			/* -y */
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
  params.use_gpu_send = 1;				/* D D */
  params.use_gpu_recv = 1;
#else
  params.use_gpu_send = 0;				/* H H */
  params.use_gpu_recv = 0;
#endif

  /* get parameters from command line options */
  while ((option = getopt(argc, argv, optlist)) != -1) {
    switch (option) {
      case 'a': {
	/* Note should be power-of-2 and multiple of sizeof(void*).
         * These are not checked here, but will be by posix_memalign() */
        params.align = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'A': {
        params.use_misaligned_data = 1;
        break;
      }
      case 'b': {
	/* I just mask down to a byte value without checking its range. */
        params.initial_byte = mystrtoul(optarg, NULL, 0) & 0xff;
        break;
      }
      case 'c' : {
        params.use_crc_check = 1;
        break;
      }
      case 'd' : {
        params.data_check = 1;
        break;
      }
      case 'D': {
        params.max_data = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'e': {
        params.use_random_length = 1;
        break;
      }
      case 'E': {
        params.max_errors = mystrtol(optarg, NULL, 0);
        break;
      }
      case 'g': {
        params.grid_ndims = mystrtol(optarg, NULL, 0);
	params.periodic_grid = 0;
	break;
      }
      case 'G': {
        params.grid_ndims = mystrtol(optarg, NULL, 0);
	params.periodic_grid = 1;
	break;
      }
      case 'h' : {
	if (comm_rank == 0) {
          get_help(argc, argv);
	}
        run = 0;
	break;
      }
      case 'i': {
        params.include_shm = 1;
        break;
      }
      case 'I': {
        params.incr_msg_size = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'l': {
        params.min_msg_size = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'L': {
        params.min_msgs = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'm': {
        params.max_msg_size = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'M': {
        params.max_msgs = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'n': {
        params.repeats = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'O' : {
        params.show_options = 1;
        break;
      }
      case 'p' : {
        params.progress = 1;
        break;
      }
      case 'P' : {
        params.poison = 1;
        break;
      }
      case 'q' : {
        params.quiet = 1;
        break;
      }
      case 'r': {
        params.use_random_data = 1;
        break;
      }
      case 'R': {
        params.round_robin = 1;
        break;
      }
      case 's': {
        params.include_self = 1;
        break;
      }
      case 'S': {
        params.mpi_send_fn = MPI_Issend;
        params.mpi_send_fn_name = "MPI_Issend";
        break;
      }
      case 't': {
        params.runtime_mins = mystrtoul(optarg, NULL, 0);
	params.repeats = UINT32_MAX;
        break;
      }
      case 'u': {
	params.uni_grid = 1;
	break;
      }
      case 'v' : {
	params.verbose++;
	break;
      }
      case 'w': {
        params.window_size = mystrtoul(optarg, NULL, 0);
        break;
      }
      case 'W': {
        params.wait_on_start = mystrtoul(optarg, NULL, 0);
	params.wait_on_exit = params.wait_on_start;
        break;
      }
      case 'x' : {
        params.use_xor_check = 1;
        break;
      }
      case 'y': {
        params.use_small_messages = 1;
        break;
      }
      case 'z': {
        params.data_check = 1;
        params.use_random_data = 1;
        params.use_xor_check = 1;
        break;
      }
      case 'Z' : {
        params.zero = 1;
        break;
      }
      default: {
        if (comm_rank == 0) {
          if (option == '?') {
            fprintf(stderr, "Error: option '%c' is invalid", optopt);
          }
          else if (option == ':') {
            fprintf(stderr, "Error: missing option argument");
          }
          else {
            fprintf(stderr, "Error: cannot process options (option=%d) ",
		    (int) option);
          }
          fprintf(stderr, ", rerun with -h for help\n");
        }
        result = EXIT_FAILURE;
	run = 0;
        break;
      }
    }
  }
  if (optind < argc) {
    if (0 == strcmp(argv[optind], "H"))
      params.use_gpu_send = 0;
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
    else if (0 == strcmp(argv[optind], "D"))
      params.use_gpu_send = 1;
#endif
    else {
      if (comm_rank == 0) {
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
        fprintf(stderr, "Error: Invalid send device '%s', must be 'H' or 'D\n", argv[optind]);
#else
        fprintf(stderr, "Error: Invalid send device '%s', must be 'H'\n", argv[optind]);
#endif
      }
      run = 0;
    }
    optind++;
  }
  if (optind < argc) {
    if (0 == strcmp(argv[optind], "H"))
      params.use_gpu_recv = 0;
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
    else if (0 == strcmp(argv[optind], "D"))
      params.use_gpu_recv = 1;
#endif
    else {
      if (comm_rank == 0) {
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
        fprintf(stderr, "Error: Invalid recv device '%s', must be 'H' or 'D\n", argv[optind]);
#else
        fprintf(stderr, "Error: Invalid recv device '%s', must be 'H'\n", argv[optind]);
#endif
      }
      run = 0;
    }
    optind++;
  }
  if (optind < argc) {
    if (comm_rank == 0) {
      fprintf(stderr, "Error: Unexpected extra arguments\n");
    }
    run = 0;
  }
  if (params.grid_ndims < 0) {
    if (comm_rank == 0) {
      fprintf(stderr, "Error: grid dimensions must not be less than 0\n");
    }
    run = 0;
  }
  if (params.min_msg_size < (uint32_t) DEFAULT_MIN_MSG_SIZE) {
    if (comm_rank == 0) {
      fprintf(stderr, "Error: minimum message size must be at least %d bytes "
	    "because of message header\n",
            DEFAULT_MIN_MSG_SIZE);
    }
    run = 0;
  }
  if (params.min_msg_size > params.max_msg_size) {
    if (comm_rank == 0) {
      fprintf(stderr, "Error: min message size must be <= than max msg size\n");
    }
    run = 0;
  }
  if (params.incr_msg_size < 0) {
    if (comm_rank == 0) {
      fprintf(stderr, "Error: message size increment must not be < 0\n");
    }
    run = 0;
  }
  if (comm_size == 1 && !params.include_self) {
    get_help(argc, argv);
    fprintf(stderr, "NOTE: Run on more than 1 process or -s to test self.\n");
    run = 0;
  }
  if (params.poison && params.zero) {
    if (comm_rank == 0) {
      fprintf(stderr, "Error: poison and zero are mutually exclusive\n");
    }
    run = 0;
  }
  if (comm_rank == 0 && params.show_options) {
    printf("Option verbose = %d\n", params.verbose);
    printf("Option progress = %d\n", params.progress);
    printf("Option show_options = %d\n", params.show_options);
    printf("Option poison = %d\n", params.poison);
    printf("Option zero = %d\n", params.zero);
    printf("Option quiet = %d\n", params.quiet);
    printf("Option include_self = %d\n", params.include_self);
    printf("Option include_shm = %d\n", params.include_shm);
    printf("Option use_crc_check = %d\n", params.use_crc_check);
    printf("Option use_xor_check = %d\n", params.use_xor_check);
    printf("Option data_check = %d\n", params.data_check);
    printf("Option use_random_data = %d\n", params.use_random_data);
    printf("Option use_misaligned_data = %d\n", params.use_misaligned_data);
    printf("Option initial_byte = %d\n", params.initial_byte);
    printf("Option round_robin = %d\n", params.round_robin);
    printf("Option grid_ndims = %d\n", params.grid_ndims);
    printf("Option periodic_grid = %d\n", params.periodic_grid);
    printf("Option uni_grid = %d\n", params.uni_grid);
    printf("Option use_random_length = %d\n", params.use_random_length);
    printf("Option repeats = %u\n", (unsigned) params.repeats);
    printf("Option runtime_mins = %u\n", (unsigned) params.runtime_mins);
    printf("Option min_msg_size = %u\n", (unsigned) params.min_msg_size);
    printf("Option max_msg_size = %u\n", (unsigned) params.max_msg_size);
    printf("Option incr_msg_size = %u\n", (unsigned) params.incr_msg_size);
    printf("Option min_msgs = %u\n", (unsigned) params.min_msgs);
    printf("Option max_msgs = %u\n", (unsigned) params.max_msgs);
    printf("Option max_data = %u\n", (unsigned) params.max_data);
    printf("Option window_size = %u\n", (unsigned) params.window_size);
    printf("Option align = %u\n", (unsigned) params.align);
    printf("Option mpi_send_fn_name = %s\n", params.mpi_send_fn_name);
    printf("Option max_errors = %d\n", params.max_errors);
  }
  if (!run) {
    MPI_Finalize();
    exit(result);
  }
  return params;
}

int main (int argc, char **argv)
{
  params_t params;
  int comm_rank, comm_size;
  int total_errors = 0;
  uint32_t r;
  int running;
  double start, end, now;

  DEFAULT_MIN_MSG_SIZE = ceilpow2(sizeof(msg_hdr_t));

#ifdef MPI_STRESS_CUDA
  int rank, num_devices, dev_id;
  char * rank_str;
  struct cudaDeviceProp dev_prop;
  CUresult curesult = CUDA_SUCCESS;
  CUdevice cuDevice;
  CUcontext cuContext;

  if ((rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL)
    rank = atoi(rank_str);
  else if ((rank_str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL)
    rank = atoi(rank_str);

  /* CUDA INIALIZATION */
  CUDA_CALL(cudaGetDeviceCount, &num_devices);
  if (num_devices < 1) {
    fprintf(stderr, "Need at least one CUDA device.\n");
    exit(1);
  }

  int i, devCnt = 0;
  int devAvail[MAX_DEVICES];
  if (num_devices > MAX_DEVICES) {
    fprintf(stderr, "Number of CUDA devices narrowed down from %d to %d\n", num_devices, MAX_DEVICES),
    num_devices = MAX_DEVICES;
  }
  for (i = 0; i < num_devices; i++)
  {
    CUDA_CALL(cudaGetDeviceProperties, &dev_prop, i);
    if (dev_prop.unifiedAddressing != 1) {
      fprintf(stderr, "Skipping CUDA device %d as it does not support Unified Virtual Addressing.\n", i);	/* Just a warning message */
      continue;
    }
    if (dev_prop.computeMode != cudaComputeModeDefault) {
      fprintf(stderr, "Skipping CUDA device %d as it is not in Compute Mode Default.\n", i);	/* Just a warning message */
      continue;
    }
    devAvail[devCnt] = i;	/* Store suitable device id */
    devCnt += 1;
  }
  if (devCnt == 0) {
    fprintf(stderr, "There is no suitable CUDA device (0/%d).\n", num_devices);
    exit(1);
  }
  i = rank % devCnt;		/* Get device id based on rank and number of suitable devices */
  dev_id = devAvail[i];

  curesult = cuInit(0);
  if (curesult != CUDA_SUCCESS) {
    exit(1);
  }

  curesult = cuDeviceGet(&cuDevice, dev_id);
  if (curesult != CUDA_SUCCESS) {
    exit(1);
  }

  curesult = cuCtxCreate(&cuContext, 0, cuDevice);
  if (curesult != CUDA_SUCCESS) {
    exit(1);
  }
#endif

#ifdef MPI_STRESS_ONEAPI
  MPI_STRESS_ONEAPI_ZE_CALL(zeInit, ZE_INIT_FLAG_GPU_ONLY);

  int i;
  int devCnt;
  int rank = 0;
  char *rank_str;
  uint32_t ze_driver_count = 1, ze_device_count = 1;

  if ((rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL)
    rank = atoi(rank_str);
  else if ((rank_str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL)
    rank = atoi(rank_str);
  else if ((rank_str = getenv("MPI_LOCALRANKID")) != NULL)
    rank = atoi(rank_str);
  else if ((rank_str = getenv("PMI_RANK")) != NULL)
    rank = atoi(rank_str);

  MPI_STRESS_ONEAPI_ZE_CALL(zeDriverGet, &ze_driver_count, &ze_driver);

  ze_context_desc_t ctxtDesc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, NULL, 0 };
  MPI_STRESS_ONEAPI_ZE_CALL(zeContextCreate, ze_driver, &ctxtDesc, &ze_context);

  ze_device_count = 0;
  MPI_STRESS_ONEAPI_ZE_CALL(zeDeviceGet, ze_driver, &ze_device_count, NULL);
  if (ze_device_count > MAX_ZE_DEVICES)
    ze_device_count = MAX_ZE_DEVICES;
  MPI_STRESS_ONEAPI_ZE_CALL(zeDeviceGet, ze_driver, &ze_device_count, devices);

  for (i = 0; i < ze_device_count; i++)
    oneapi_ze_cmd_create(devices[i], &ze_devices[i]);

  devCnt = ze_device_count;
  srandom((unsigned int)time(NULL));
  i = random() % devCnt;
  cur_ze_dev = &ze_devices[i];
  ze_cq = cur_ze_dev->cq;
  ze_cl = cur_ze_dev->cl;
#endif

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  /* MPI_Get_processor_name(myhostname, &r); */
  gethostname(myhostname, MPI_MAX_PROCESSOR_NAME);

  params = get_params(argc, argv, comm_rank, comm_size);
#if defined(MPI_STRESS_CUDA)
  printf("Rank %d: Using Cuda Device %d (%d total)\n", comm_rank, i, devCnt);
#elif defined(MPI_STRESS_ONEAPI)
  printf("Rank %d: Using OneAPI Device %d (%d total)\n", comm_rank, i, devCnt);
#endif

  if (params.wait_on_start) {
    if (comm_rank == 0) {
      printf("Wait for %d seconds at start\n", params.wait_on_start);
    }
    printf("Rank %d is running as PID %d on host %s\n", comm_rank, getpid(),
	   myhostname);
    sleep(params.wait_on_start);
  }

  if (comm_rank == 0) {
    time_t t = time(NULL);
    char s[26];
    ctime_r(&t, s);
    printf("Start mpi_stress at %s", s);
#if defined(MPI_STRESS_CUDA) || defined(MPI_STRESS_ONEAPI)
    printf("Send Buffer on %s and Receive Buffer on %s\n",
	   params.use_gpu_send ? "DEVICE (D)" : "HOST (H)",
	   params.use_gpu_recv ? "DEVICE (D)" : "HOST (H)");
#endif
  }

  r = 0;
  start = MPI_Wtime();
  now = MPI_Wtime();
  end = start + (60.0 * params.runtime_mins);

  running = (r < params.repeats);
  while (running) {
    int errors = run(params, comm_rank, comm_size, 0xCABBA6E5 + comm_rank);
    int my_running;
    total_errors += errors;
    now = MPI_Wtime();
    if (comm_rank == 0) {
      time_t t = time(NULL);
      char s[26];
      char *p;
      ctime_r(&t, s);
      p = index(s, '\n');
      if (p != NULL) {
        *p = 0;
      }
      printf("\nIteration %d : errors = %d, total = %d (%d secs, %s)\n", 
             r, errors, total_errors, (int) (now - start), s);
    }
    r++;
    my_running = (r < params.repeats) && (!params.runtime_mins || now < end);
    MPI_Allreduce(&my_running, &running, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }

  if (comm_rank == 0) {
    int mins = (int) ((now - start) / 60.0);
    int secs = (int) ((now - start) - (mins * 60.0));
    printf("After %d iteration(s), %d mins and %d secs, "
           "total errors = %d\n",
           (int) r, mins, secs, total_errors);
  }

  if (params.wait_on_exit) {
    if (comm_rank == 0) {
      printf("Wait for %d seconds at exit\n", params.wait_on_exit);
    }
    sleep(params.wait_on_exit);
  }

  MPI_Finalize();
#ifdef MPI_STRESS_CUDA
  curesult = cuCtxDestroy(cuContext);
  if (curesult != CUDA_SUCCESS) {
    exit(1);
  }
#endif
#ifdef MPI_STRESS_ONEAPI
  MPI_STRESS_ONEAPI_ZE_CALL(zeCommandListDestroy, ze_cl);
  MPI_STRESS_ONEAPI_ZE_CALL(zeCommandQueueDestroy, ze_cq);
  MPI_STRESS_ONEAPI_ZE_CALL(zeContextDestroy, ze_context);
#endif

  return total_errors ? EXIT_FAILURE : EXIT_SUCCESS;
}
