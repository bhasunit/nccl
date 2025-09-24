/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include "debug.h"
#include "nccl_tuner.h"
#include "checks.h"

static ncclTuner_v5_t* ncclTuner_v5;
static ncclTuner_t ncclTuner;

static ncclResult_t ncclTuner_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes, int numPipeOps, float** collCostTable, int numAlgo, int numProto, int regBuff, int* nChannels) {
  NCCLCHECK(ncclTuner_v5->getCollInfo(context, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto, regBuff, nChannels));
  return ncclSuccess;
}

static ncclResult_t ncclTuner_finalize(void* ctx) {
  return ncclTuner_v5->finalize(ctx);
}

static ncclResult_t ncclTuner_init(void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logfn,
                                   ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_t* constants) {
  ncclTunerConstants_v5_t v5_constants;

  memcpy(&v5_constants, constants, sizeof(v5_constants));
  NCCLCHECK(ncclTuner_v5->init(context, commId, nRanks, nNodes, logfn, nvlDomainInfo, &v5_constants));
  ncclTuner.getCollInfo = ncclTuner_getCollInfo;
  ncclTuner.finalize = ncclTuner_finalize;
  return ncclSuccess;
}

ncclTuner_t* getNcclTuner_v5(void* lib) {
  ncclTuner_v5 = (ncclTuner_v5_t*)dlsym(lib, "ncclTunerPlugin_v5");
  if (ncclTuner_v5) {
    ncclTuner.name = ncclTuner_v5->name;
    ncclTuner.init = ncclTuner_init;
    INFO(NCCL_INIT|NCCL_TUNING, "TUNER/Plugin: Using %s (v5)", ncclTuner_v5->name);
    return &ncclTuner;
  }
  return NULL;
}
