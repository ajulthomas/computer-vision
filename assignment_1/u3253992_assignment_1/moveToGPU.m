% moveToGPU.m
function dataOut = moveToGPU(dataIn, useGPU)
% MOVETOGPU Moves data to GPU if useGPU is true, otherwise returns input.

    if useGPU
        dataOut = gpuArray(dataIn);
    else
        dataOut = dataIn;
    end
end
