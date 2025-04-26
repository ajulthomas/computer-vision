% gatherFromGPU.m
function dataOut = gatherFromGPU(dataIn, useGPU)
% GATHERFROMGPU Gathers data from GPU if useGPU is true, otherwise returns input.

    if useGPU
        dataOut = gather(dataIn);
    else
        dataOut = dataIn;
    end
end
