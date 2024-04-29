simion.workbench_program()
-- Set scale for the first PA instance in the workbench.
-- Replace '0.05' with the desired scale factor.
adjustable scale_factor = 0.05
-- Assuming there is at least one PA instance loaded.
local pa_instance = simion.wb.instances[1]
if pa_instance ~= nil then
  pa_instance.scale = scale_factor
end