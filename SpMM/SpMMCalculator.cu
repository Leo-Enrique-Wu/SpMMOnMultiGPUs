#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <omp.h>

#define TILE_WIDTH_K 32
#define TILE_WIDTH_N 32

template <class T>
void print1dArr(T * arr, int num, bool printDetail);

template <class T>
void print2dArr(T * arr, int rowNum, int colNum, bool printDetail);

void printToFile(float * arr, int rowNum, int colNum, FILE * fp);

__global__
void spMM(int subDMatColIdxOffset, int * spMatIndptrs, int * spMatIndices, float * spMatData, float * subDMat,
          float * subOutputMat, int n, int subN) {

        __shared__ int spMatIndicesTile[TILE_WIDTH_K];
        __shared__ float spMatDataTile[TILE_WIDTH_K];
        __shared__ float dMatDataTile[TILE_WIDTH_K][TILE_WIDTH_N];
        // __shared__ int outputDataTile[TILE_WIDTH_N];
        float outputData = 0.0;

        int rowIdx = blockIdx.x;
        int relativeColIdx = blockIdx.y * blockDim.y + threadIdx.y;
        int colIdx = subDMatColIdxOffset + relativeColIdx;
        int nnzNum = spMatIndptrs[rowIdx + 1] - spMatIndptrs[rowIdx];
        int spMatIndicesStartIdx = spMatIndptrs[rowIdx];
        for ( ; nnzNum > 0; nnzNum -= TILE_WIDTH_K) {

                int spMatIndicesIdx = spMatIndicesStartIdx + threadIdx.x;
                if (threadIdx.y == 0) {
                        if (threadIdx.x < nnzNum) {
                                spMatIndicesTile[threadIdx.x] = spMatIndices[spMatIndicesIdx];
                                spMatDataTile[threadIdx.x] = spMatData[spMatIndicesIdx];
                        }
                }
                __syncthreads();

                if (threadIdx.x < nnzNum && colIdx < n) {
                        int subDMatIdx = spMatIndicesTile[threadIdx.x] * subN + relativeColIdx;
                        dMatDataTile[threadIdx.x][threadIdx.y] = subDMat[subDMatIdx] * spMatDataTile[threadIdx.x];
                } else {
                        dMatDataTile[threadIdx.x][threadIdx.y] = 0.0;
                }
                __syncthreads();

                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                        if (threadIdx.x < stride) {
                                dMatDataTile[threadIdx.x][threadIdx.y] += dMatDataTile[threadIdx.x + stride][threadIdx.y];
                        }
                        __syncthreads();
                }

                if (threadIdx.x == 0) {
                        outputData += dMatDataTile[threadIdx.x][threadIdx.y];
                }

                spMatIndicesStartIdx += TILE_WIDTH_K;
                __syncthreads();

        }

        if (threadIdx.x == 0) {
                int subOutputMatIdx = rowIdx * subN + relativeColIdx;
                subOutputMat[subOutputMatIdx] = outputData;
        }

}

// g++ -std=c++11 -lm -Wall -o SpMM ./SpMMCalculator.cpp
int main(int argc, char * argv[]) {

        cudaError_t theErr;
        double start, end;

        clock_t startCt, endCt;
        double elaspedTime;

        // printf("Input Param:\n");
        int m = std::atoi(argv[1]);
        int k = std::atoi(argv[2]);
        int n = std::atoi(argv[3]);
        // printf("M = %d, K = %d, N = %d\n", m, k, n);

        char * inputSpMatFileName = argv[4];
        char * inputDMatFileName = argv[5];
        // char * outputFileName = argv[6];
        // printf("inputSpMatFileName = %s\n", inputSpMatFileName);
        // printf("inputDMatFileName = %s\n", inputDMatFileName);
        // printf("outputFileName = %s\n", outputFileName);

        int isPrintOutput = 0;
        if (argc >= 7) {
                isPrintOutput = std::atoi(argv[6]);
        }

        int divCombineThreadNum = 4;
        if (argc >= 8) {
                divCombineThreadNum = std::atoi(argv[7]);
        }

        int executeType = 1;
        // 0: Serial
        // 1: GPU
        if (argc >= 9) {
                executeType = std::atoi(argv[8]);
        }

        int gpuNum = 2;
        if (argc >= 10) {
                gpuNum = std::atoi(argv[9]);
        }

        float * outputData = (float *)malloc(m * n * sizeof(float));

        // Read matrices from the input files
        // 1. The sparse matrix
        // 2. The dense matrix
        std::string line;
        std::ifstream inputSpMatFileStream (inputSpMatFileName);
        if (!inputSpMatFileStream.is_open()) {
                fprintf(stderr, "Failed to open file: %s\n", inputSpMatFileName);
                exit(1);
        }

        // printf("Reading the inptut matrices\n");
        startCt = clock();

        int nnzNum = 0;
        int spMatIndptrsSize = m + 1;
        int * spMatIndptrs = (int *)malloc(spMatIndptrsSize * sizeof(int));
        int * spMatIndices;
        float * spMatData;
        const char * lineIdx;
        char intStr[10];
        char floatStr[50];
        std::string::size_type startIdx;
        std::string::size_type nextDelimIdx;
        int type = 1;
        int idxOffset = 0;
        int itemNumPerLine = 10;
        int restItemNum = 0;
        int itemNum = 0;

        while (getline(inputSpMatFileStream, line)) {

                switch (type) {
                case 1:
                        startIdx = 0;
                        lineIdx = line.c_str();

                        restItemNum = spMatIndptrsSize - idxOffset;
                        itemNum = (restItemNum < itemNumPerLine)? restItemNum : itemNumPerLine;

                        for (int i = 0; i < itemNum; ++i) {
                                int spMatIndptrsIdx = idxOffset + i;
                                sscanf(lineIdx, "%d", &spMatIndptrs[spMatIndptrsIdx]);
                                sprintf(intStr, "%d", spMatIndptrs[spMatIndptrsIdx]);
                                nextDelimIdx = line.find(intStr, startIdx);
                                nextDelimIdx = line.find(" ", nextDelimIdx);
                                lineIdx = lineIdx + (nextDelimIdx - startIdx);
                                startIdx = nextDelimIdx;
                        }

                        idxOffset += itemNum;
                        if (idxOffset == spMatIndptrsSize) {
                                // printf("Finished processing spMatIndptrs\n");
                                type = 2;
                                idxOffset = 0;
                        }
                        break;
                case 2:
                        sscanf(line.c_str(), "%d", &nnzNum);

                        spMatIndices = (int *)malloc(nnzNum * sizeof(int));
                        spMatData = (float *)malloc(nnzNum * sizeof(float));
                        // printf("# of nnz: %d\n", nnzNum);
                        type = 3;
                        break;
                case 3:
                        startIdx = 0;
                        lineIdx = line.c_str();

                        restItemNum = nnzNum - idxOffset;
                        itemNum = (restItemNum < itemNumPerLine)? restItemNum : itemNumPerLine;

                        for (int i = 0; i < itemNum; ++i) {
                                int spMatIndicesIdx = idxOffset + i;
                                sscanf(lineIdx, "%d", &spMatIndices[spMatIndicesIdx]);
                                sprintf(intStr, "%d", spMatIndices[spMatIndicesIdx]);
                                nextDelimIdx = line.find(intStr, startIdx);
                                nextDelimIdx = line.find(" ", nextDelimIdx);
                                lineIdx = lineIdx + (nextDelimIdx - startIdx);
                                startIdx = nextDelimIdx;
                        }

                        idxOffset += itemNum;
                        if (idxOffset == nnzNum) {
                                // printf("Finished processing spMatIndices\n");
                                type = 4;
                                idxOffset = 0;
                        }
                        break;
                case 4:
                        startIdx = 0;
                        lineIdx = line.c_str();

                        /*
                           if (idxOffset == 10570) {
                                printf("idxOffset=%d, line=%s\n", idxOffset, lineIdx);
                           }
                         */

                        restItemNum = nnzNum - idxOffset;
                        itemNum = (restItemNum < itemNumPerLine)? restItemNum : itemNumPerLine;

                        for (int i = 0; i < itemNum; ++i) {

                                int spMatDataIdx = idxOffset + i;
                                /*
                                   if (idxOffset == 10570) {
                                        printf("idxOffset=%d, i=%d, line=%s\n", idxOffset, i, lineIdx);
                                   }
                                 */

                                sscanf(lineIdx, "%f", &spMatData[spMatDataIdx]);
                                sprintf(floatStr, "%f", spMatData[spMatDataIdx]);
                                /*
                                   if (idxOffset == 10570 && i == 6) {
                                        printf("%f -> %s\n", spMatData[spMatDataIdx], floatStr);
                                   }
                                 */
                                // printf("%f -> %s\n", spMatData[i], floatStr);
                                std::string fstr(floatStr);
                                size_t fsubstrlen;
                                do {
                                        fsubstrlen = fstr.length() - 1;
                                        // printf("fstr=%s, fstr.length()=%lu\n", fstr.c_str(), fstr.length());
                                        fstr = fstr.substr(0, fsubstrlen);
                                        // printf("fstr.substr=%s\n", fstr.c_str());
                                        nextDelimIdx = line.find(fstr, startIdx);
                                } while (nextDelimIdx == std::string::npos && fsubstrlen > 1);
                                /*
                                   if (nextDelimIdx == std::string::npos) {
                                        printf("idxOffset=%d, i=%d, spMatData[%d]: nextDelimIdx == std::string::npos\n",
                                               idxOffset, i, spMatDataIdx);
                                        printf("line=%s, finding float: %f\n", lineIdx, spMatData[spMatDataIdx]);
                                   }
                                 */
                                // printf("nextDelimIdx(%s)=%lu\n", fstr.c_str(), nextDelimIdx);
                                nextDelimIdx = line.find(" ", nextDelimIdx);
                                // printf("nextDelimIdx(space)=%lu\n", nextDelimIdx);
                                lineIdx = lineIdx + (nextDelimIdx - startIdx);
                                startIdx = nextDelimIdx;

                        }

                        idxOffset += itemNum;
                        if (idxOffset == nnzNum) {
                                // printf("Finished processing spMatData\n");
                                type = 5;
                                idxOffset = 0;
                        }
                        break;
                default:
                        printf("%s\n", line.c_str());
                }
        }
        inputSpMatFileStream.close();

        /*
           for (int i = 1861; i < 1871; ++i) {
                printf("spMatData[%d]=%f\n", i, spMatData[i]);
           }
         */

        /*
           printf("spMatIndptrs:\n");
           print1dArr<int>(spMatIndptrs, spMatIndptrsSize, true);
           printf("spMatIndices:\n");
           print1dArr<int>(spMatIndices, nnzNum, true);
           printf("spMatData:\n");
           print1dArr<float>(spMatData, nnzNum, true);
         */

        // printf("Reading a dense matrix\n");
        int dMatEleNum = k * n;
        float * dMatData = (float *)malloc(dMatEleNum * sizeof(float));
        // printf("size of dMatData: %d\n", dMatEleNum);

        std::ifstream inputDMatFileStream (inputDMatFileName);
        if (!inputDMatFileStream.is_open()) {
                fprintf(stderr, "Failed to open file: %s\n", inputDMatFileName);
                exit(1);
        }

        int rowIdx = 0;
        while (getline(inputDMatFileStream, line)) {
                // printf("Processing rowIdx=%d\n", rowIdx);
                startIdx = 0;
                lineIdx = line.c_str();

                restItemNum = n - idxOffset;
                itemNum = (restItemNum < itemNumPerLine)? restItemNum : itemNumPerLine;

                for (int i = 0; i < itemNum; ++i) {
                        int colIdx = idxOffset + i;
                        int dMatDataIdx = rowIdx * n + colIdx;
                        // printf("i=%d, dMatDataIdx=%d\n", i, dMatDataIdx);
                        sscanf(lineIdx, "%f", &dMatData[dMatDataIdx]);
                        sprintf(floatStr, "%f", dMatData[dMatDataIdx]);
                        // printf("%f -> %s\n", spMatData[i], floatStr);
                        std::string fstr(floatStr);
                        size_t fsubstrlen;
                        do {
                                fsubstrlen = fstr.length() - 1;
                                // printf("fstr=%s, fstr.length()=%lu\n", fstr.c_str(), fstr.length());
                                fstr = fstr.substr(0, fsubstrlen);
                                // printf("fstr.substr=%s\n", fstr.c_str());
                                nextDelimIdx = line.find(fstr, startIdx);
                        } while (nextDelimIdx == std::string::npos && fsubstrlen > 1);
                        // printf("nextDelimIdx(%s)=%lu\n", fstr.c_str(), nextDelimIdx);
                        nextDelimIdx = line.find(" ", nextDelimIdx);
                        // printf("nextDelimIdx(space)=%lu\n", nextDelimIdx);
                        lineIdx = lineIdx + (nextDelimIdx - startIdx);
                        startIdx = nextDelimIdx;
                }

                idxOffset += itemNum;
                if (idxOffset == n) {
                        // printf("Finished processing dData[rowIdx=%d]\n", rowIdx);
                        ++rowIdx;
                        idxOffset = 0;
                }

        }
        inputDMatFileStream.close();
        endCt = clock();
        elaspedTime = ((double)(endCt - startCt)) / CLOCKS_PER_SEC;
        printf("Finished reading the input matrices, elaspedTime=%f\n", elaspedTime);

        // printf("dMatData:\n");
        // print2dArr<float>(dMatData, k, n, true);

        if (executeType == 1) {

                int dev_cnt = 0;
                theErr = cudaGetDeviceCount(&dev_cnt);
                if(theErr != cudaSuccess)
                {
                        printf("Error: %s\n", cudaGetErrorString(theErr));
                        exit(-1);
                }

                gpuNum = (dev_cnt < gpuNum)? dev_cnt : gpuNum;
                // printf("Use # of GPUs: %d\n", gpuNum);

                // Devided the dense matrix into `gpuNum` submatrices
                float * subDMats[gpuNum];
                float * subOutputMats[gpuNum];
                int subN = (n + gpuNum - 1) / gpuNum;
                int subDMatSize = k * subN;

                if (gpuNum > 1) {
                        printf("divCombineThreadNum=%d\n", divCombineThreadNum);
                        start = omp_get_wtime();
                        for (int devId = 0; devId < gpuNum; ++devId) {

                                float * subDMat = (float *)malloc(subDMatSize * sizeof(float));
                                // printf("Allocate subDMat for device#%d\n", devId);
                                subDMats[devId] = subDMat;
                                float * subOutputMat = (float *)malloc(subDMatSize * sizeof(float));
                                // printf("Allocate subOutputMat for device#%d\n", devId);
                                subOutputMats[devId] = subOutputMat;

                                int subDMatColStartIdx = devId * subN;
                                int subDMatColExcludeEndIdx = (devId + 1) * subN;

                        #pragma omp parallel for num_threads(divCombineThreadNum)
                                for (int rolIdx = 0; rolIdx < k; ++rolIdx) {
                                        for (int colIdx = subDMatColStartIdx; colIdx < subDMatColExcludeEndIdx; ++colIdx) {
                                                int subDMatIdx = rolIdx * subN + (colIdx - subDMatColStartIdx);
                                                int dMatIdx = rolIdx * n + colIdx;
                                                subDMat[subDMatIdx] = dMatData[dMatIdx];
                                        }

                                }
                                // printf("subDMat for device#%d:\n", devId);
                                // print2dArr<float>(subDMat, k, subN, true);
                                // printf("Finished device#%d\n", devId);

                        }
                        end = omp_get_wtime();
                        printf("Elapsed time for deviding the dense matrix: %f\n", (end - start));
                }

                // Allocate memory on the GPU devices
                // And, transfer the sparse matrix and the dense submatrices to GPUs
                // Launch the kernel
                if (gpuNum > 1) {
                        start = omp_get_wtime();
                        #pragma omp parallel num_threads(gpuNum)
                        {
                                int * spMatIndptrsArr_d[gpuNum];
                                int * spMatIndicesArr_d[gpuNum];
                                float * spMatDataArr_d[gpuNum];
                                float * subDMatArr_d[gpuNum];
                                float * subOutputMats_d[gpuNum];

                                #pragma omp for
                                for (int devId = 0; devId < gpuNum; ++devId) {

                                        // int myRank = omp_get_thread_num();
                                        // printf("[t-%d] use GPU device#%d\n", myRank, devId);
                                        cudaSetDevice(devId);

                                        int * spMatIndptrs_d;
                                        int * spMatIndices_d;
                                        float * spMatData_d;
                                        float * subDMat_d;
                                        float * subOutputMat_d;

                                        cudaError_t err = cudaMalloc((void **) &spMatIndptrs_d, spMatIndptrsSize * sizeof(int));
                                        if (err != cudaSuccess) {
                                                fprintf(stderr, " Cannot allocate spMatIndptrs_d on device\n");
                                                // return 1;
                                        }

                                        err = cudaMalloc((void **) &spMatIndices_d, nnzNum * sizeof(int));
                                        if (err != cudaSuccess) {
                                                fprintf(stderr, " Cannot allocate spMatIndices_d on device\n");
                                                // return 1;
                                        }

                                        err = cudaMalloc((void **) &spMatData_d, nnzNum * sizeof(float));
                                        if (err != cudaSuccess) {
                                                fprintf(stderr, " Cannot allocate spMatData_d on device\n");
                                                // return 1;
                                        }

                                        err = cudaMalloc((void **) &subDMat_d, subDMatSize * sizeof(float));
                                        if (err != cudaSuccess) {
                                                fprintf(stderr, " Cannot allocate subDMat_d on device\n");
                                                // return 1;
                                        }

                                        err = cudaMalloc((void **) &subOutputMat_d, subDMatSize * sizeof(float));
                                        if (err != cudaSuccess) {
                                                fprintf(stderr, " Cannot allocate subOutputMat_d on device\n");
                                                // return 1;
                                        }

                                        spMatIndptrsArr_d[devId] = spMatIndptrs_d;
                                        spMatIndicesArr_d[devId] = spMatIndices_d;
                                        spMatDataArr_d[devId] = spMatData_d;
                                        subDMatArr_d[devId] = subDMat_d;
                                        subOutputMats_d[devId] = subOutputMat_d;

                                        // Transfer data to the GPU device
                                        float * subDMat = subDMats[devId];
                                        cudaMemcpy(spMatIndptrs_d, spMatIndptrs, spMatIndptrsSize * sizeof(int), cudaMemcpyHostToDevice);
                                        cudaMemcpy(spMatIndices_d, spMatIndices, nnzNum * sizeof(int), cudaMemcpyHostToDevice);
                                        cudaMemcpy(spMatData_d, spMatData, nnzNum * sizeof(float), cudaMemcpyHostToDevice);
                                        cudaMemcpy(subDMat_d, subDMat, subDMatSize * sizeof(float), cudaMemcpyHostToDevice);

                                        //   __global__
                                        // void spMM(int subDMatColIdxOffset, int * spMatIndptrs, int * spMatIndices, float * spMatData, float * subDMat,
                                        //        float * subOutputMat, int n, int subN)

                                        int subDMatColIdxOffset = devId * subN;
                                        size_t gridDimY = (subN + TILE_WIDTH_N - 1) / TILE_WIDTH_N;
                                        dim3 dimGrid(m, gridDimY);
                                        dim3 dimBlock(TILE_WIDTH_K, TILE_WIDTH_N);
                                        spMM<<<dimGrid, dimBlock>>>(subDMatColIdxOffset, spMatIndptrs_d, spMatIndices_d, spMatData_d, subDMat_d,
                                                                    subOutputMat_d, n, subN);


                                }

                                // Transfer the output submatrices back to the host
                                #pragma omp for
                                for (int devId = 0; devId < gpuNum; ++devId) {

                                        // int myRank = omp_get_thread_num();
                                        // printf("[t-%d] use GPU device#%d\n", myRank, devId);
                                        cudaSetDevice(devId);

                                        float * subOutputMat = subOutputMats[devId];
                                        float * subOutputMat_d = subOutputMats_d[devId];
                                        cudaMemcpy(subOutputMat, subOutputMat_d, subDMatSize * sizeof(float), cudaMemcpyDeviceToHost);

                                        cudaFree(spMatIndptrsArr_d[devId]);
                                        cudaFree(spMatIndicesArr_d[devId]);
                                        cudaFree(spMatDataArr_d[devId]);
                                        cudaFree(subDMatArr_d[devId]);
                                        cudaFree(subOutputMats_d[devId]);

                                }
                        }
                        cudaDeviceSynchronize();
                        end = omp_get_wtime();
                        printf("Elapsed time to launch the kernel: %f\n", (end - start));

                } else {

                        printf("Launch the kernel without OpenMP\n");
                        int * spMatIndptrs_d;
                        int * spMatIndices_d;
                        float * spMatData_d;
                        float * dMat_d;
                        float * outputMat_d;

                        cudaError_t err = cudaMalloc((void **) &spMatIndptrs_d, spMatIndptrsSize * sizeof(int));
                        if (err != cudaSuccess) {
                                fprintf(stderr, " Cannot allocate spMatIndptrs_d on device\n");
                                // return 1;
                        }

                        err = cudaMalloc((void **) &spMatIndices_d, nnzNum * sizeof(int));
                        if (err != cudaSuccess) {
                                fprintf(stderr, " Cannot allocate spMatIndices_d on device\n");
                                // return 1;
                        }

                        err = cudaMalloc((void **) &spMatData_d, nnzNum * sizeof(float));
                        if (err != cudaSuccess) {
                                fprintf(stderr, " Cannot allocate spMatData_d on device\n");
                                // return 1;
                        }

                        err = cudaMalloc((void **) &dMat_d, dMatEleNum * sizeof(float));
                        if (err != cudaSuccess) {
                                fprintf(stderr, " Cannot allocate subDMat_d on device\n");
                                // return 1;
                        }

                        err = cudaMalloc((void **) &outputMat_d, dMatEleNum * sizeof(float));
                        if (err != cudaSuccess) {
                                fprintf(stderr, " Cannot allocate subOutputMat_d on device\n");
                                // return 1;
                        }

                        // Transfer data to the GPU device
                        cudaMemcpy(spMatIndptrs_d, spMatIndptrs, spMatIndptrsSize * sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(spMatIndices_d, spMatIndices, nnzNum * sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(spMatData_d, spMatData, nnzNum * sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(dMat_d, dMatData, subDMatSize * sizeof(float), cudaMemcpyHostToDevice);

                        //   __global__
                        // void spMM(int subDMatColIdxOffset, int * spMatIndptrs, int * spMatIndices, float * spMatData, float * subDMat,
                        //        float * subOutputMat, int n, int subN)
                        size_t gridDimY = (subN + TILE_WIDTH_N - 1) / TILE_WIDTH_N;
                        dim3 dimGrid(m, gridDimY);
                        dim3 dimBlock(TILE_WIDTH_K, TILE_WIDTH_N);
                        spMM<<<dimGrid, dimBlock>>>(0, spMatIndptrs_d, spMatIndices_d, spMatData_d, dMat_d,
                                                    outputMat_d, n, n);

                        cudaMemcpy(outputData, outputMat_d, dMatEleNum * sizeof(float), cudaMemcpyDeviceToHost);

                        cudaFree(spMatIndptrs_d);
                        cudaFree(spMatIndices_d);
                        cudaFree(spMatData_d);
                        cudaFree(dMat_d);
                        cudaFree(outputMat_d);

                }

                // Combine the output submatrices into a final output matrix
                if (gpuNum > 1) {
                        start = omp_get_wtime();
                        for (int devId = 0; devId < gpuNum; ++devId) {

                                float * subOutputMat = subOutputMats[devId];
                                int subOutputMatColIdxOffset = devId * subN;

                        #pragma omp parallel for num_threads(divCombineThreadNum)
                                for (int rowIdx = 0; rowIdx < m; ++rowIdx) {
                                        for (int relaColIdx = 0; relaColIdx < subN && subOutputMatColIdxOffset + relaColIdx < n; ++relaColIdx) {
                                                int colIdx = subOutputMatColIdxOffset + relaColIdx;
                                                int outputDataIdx = rowIdx * n + colIdx;
                                                int subOutputDataIdx = rowIdx * subN + relaColIdx;
                                                outputData[outputDataIdx] = subOutputMat[subOutputDataIdx];
                                        }
                                }

                        }
                        end = omp_get_wtime();
                        printf("Elapsed time to combine the result: %f\n", (end - start));
                }

        } else if (executeType == 0) {

                for (int rowIdx = 0; rowIdx < m; ++rowIdx) {

                        int nnzNum = spMatIndptrs[rowIdx + 1] - spMatIndptrs[rowIdx];
                        int spMatDataIdxOffset = spMatIndptrs[rowIdx];
                        /*
                           if (rowIdx == 7) {
                                printf("rowIdx=%d, spMatDataIdxOffset=%d\n", rowIdx, spMatDataIdxOffset);
                           }
                         */

                        for (int colIdx = 0; colIdx < n; ++colIdx) {

                                int arrIdx = rowIdx * n + colIdx;
                                float value = 0.0;

                                for (int i = 0; i < nnzNum; ++i) {
                                        int spMatDataIdx = spMatDataIdxOffset + i;
                                        int dMatDataIdx = spMatIndices[spMatDataIdx] * n + colIdx;
                                        value += spMatData[spMatDataIdx] * dMatData[dMatDataIdx];
                                        /*
                                           if (rowIdx == 7 && colIdx == 0) {
                                                printf("+= (i=%d) %f, %f * %f = spMatData[%d] * dMatData[%d][%d]\n", i,
                                                       spMatData[spMatDataIdx] * dMatData[dMatDataIdx],
                                                       spMatData[spMatDataIdx], dMatData[dMatDataIdx], spMatDataIdx,
                                                       spMatIndices[spMatDataIdx], colIdx);
                                           }
                                         */
                                }
                                outputData[arrIdx] = value;

                                /*
                                   if (rowIdx >= 6 && colIdx == 0 && rowIdx < 10) {
                                        printf("nnzNum=%d, outputData[%d][%d]=outputData[%d]=%f\n", nnzNum, rowIdx, colIdx, arrIdx, outputData[arrIdx]);
                                   }
                                 */

                        }

                }

        }

        if (isPrintOutput == 1) {
                char filename[32];
                strcpy(filename, "res_");
                if (executeType == 1) {
                        strcat(filename, "g_");
                        const int n = snprintf(NULL, 0, "%d", gpuNum);
                        char buf[n+1];
                        int c = snprintf(buf, n+1, "%d", gpuNum);
                        strcat(filename, buf);
                        strcat(filename, ".txt");
                } else {
                        strcat(filename, "ref_seq.txt");
                }

                FILE * fp = fopen(filename,"w+t");
                if(!fp) {
                        printf("Cannot create file %s\n", filename);
                        exit(10);
                }
                printToFile(outputData, m, n, fp);
                fclose(fp);
                printf("Finished writing the result\n");
        }

        // Free allocated memory
        /*
           for (int devId = 0; devId < gpuNum; ++devId) {
                free(subOutputMats[devId]);
           }

           free(spMatIndptrs);
           free(spMatIndices);
           free(spMatData);
           free(dMatData);
           free(outputData);
         */

        printf("Done.\n");

}

template <class T>
void print1dArr(T * arr, int num, bool printDetail) {

        if (printDetail) {
                printf("[");
        }
        int idx = 0;
        for (int i = 0; i < num; ++i) {

                T value = arr[i];
                if (printDetail) {
                        std::cout << value << ", ";
                }
                idx++;
                if ((idx != 0) && (idx % 10) == 0) {
                        if (printDetail) {
                                printf("\n");
                        }
                }

        }
        if (printDetail) {
                printf("], total %d numbers\n", idx);
        } else {
                printf("total %d numbers\n", idx);
        }

}

template <class T>
void print2dArr(T * arr, int rowNum, int colNum, bool printDetail) {

        if (printDetail) {
                printf("[");
                for (int i = 0; i < rowNum; ++i) {
                        for (int j = 0; j < colNum; ++j) {

                                int arrIdx = i * colNum + j;
                                T value = arr[arrIdx];
                                std::cout << value;
                                if (j == colNum - 1) {
                                        printf("\n");
                                } else {
                                        std::cout << ", ";
                                }

                        }
                }
        }

        if (printDetail) {
                printf("], total %d numbers\n", rowNum * colNum);
        } else {
                printf("total %d numbers\n", rowNum * colNum);
        }

}

void printToFile(float * arr, int rowNum, int colNum, FILE * fp) {

        // printf("Solutions:\n");
        for (int i = 0; i < rowNum; ++i) {
                for (int j = 0; j < colNum; ++j) {
                        int idx = i * colNum + j;
                        fprintf(fp, "%f", arr[idx]);
                        if (j != colNum - 1) {
                                if (j % 5 != 9) {
                                        fprintf(fp, " ");
                                } else {
                                        fprintf(fp, "\n");
                                }
                        } else {
                                fprintf(fp, "\n");
                        }
                        // printf("%f\n", arr[i]);
                }
        }

}
