% enableGPU.m
function [device, useGPU] = enableGPU()
% ENABLEGPU Checks if GPU is available, resets it, and returns usage flag.
% Returns device object and useGPU flag.

    if gpuDeviceCount > 0
        device = gpuDevice(1);
        reset(device);  % Clear old data
        disp('✅ GPU found and ready:');
        disp(device);
        useGPU = true;
    else
        device = [];
        disp('⚠️ No GPU found. Running on CPU.');
        useGPU = false;
    end
end
