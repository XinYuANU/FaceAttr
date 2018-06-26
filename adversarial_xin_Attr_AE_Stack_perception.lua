require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'pl'

local adversarial = {}

function randomChange_Attr(attrs)
	local num = attrs:size(1)
	local attr_num = attrs:size(2)
	local res = attrs:clone()
	torch.setdefaulttensortype('torch.FloatTensor')
	for i = 1, num do
		if res[i][attr_num] == 0 then
			local idx = attr_num
			res[i][idx] = 1-res[i][idx]
		else
			while true do
				local idx = torch.random(1, ntrain)
				idx = torch.LongTensor{idx}
				if attrs[i] ~= dataset_Attr:index(1, idx) then
					res[i] = dataset_Attr:index(1, idx):clone()
					break
				end
			end

		end		
	end
	torch.setdefaulttensortype('torch.CudaTensor')
	return res
end

function reArrangeBatch(IDX, data_attr)
	local attrs = data_attr:index(1, IDX)
	local bs = attrs:size(1)
	local nattr = attrs:size(2)
	local IDX_Batch = IDX:clone()
	local tmp = torch.eq(attrs[{{}, nattr}], 0)
	-- print(tmp)

	if torch.any(tmp) then
		local res = IDX[tmp]
		local sz = res:size(1)
		IDX_Batch[{{1, sz}}]    = IDX[torch.eq(attrs[{{}, nattr}], 0)]:clone()
		IDX_Batch[{{sz+1, bs}}] = IDX[torch.eq(attrs[{{}, nattr}], 1)]:clone()
	end
	
	return IDX_Batch

end


function rmsprop(opfunc, x, config, state)
	
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-2
    local alpha = config.alpha or 0.9
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
        -- (2) initialize mean square values and square gradient storage
        if not state.m then
          state.m = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
          state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
        end

        -- (3) calculate new (leaky) mean squared values
        state.m:mul(alpha)
        state.m:addcmul(1.0-alpha, dfdx, dfdx)

        -- (4) perform update
        state.tmp:sqrt(state.m):add(epsilon)
        -- only opdate when optimize is true
        
        
		if config.numUpdates < 50 then
			  --io.write(" ", lr/50.0, " ")
			  x:addcdiv(-lr/50.0, dfdx, state.tmp)
		elseif config.numUpdates < 100 then
			--io.write(" ", lr/5.0, " ")
			x:addcdiv(-lr /5.0, dfdx, state.tmp)
		else 
		  --io.write(" ", lr, " ")
		  x:addcdiv(-lr, dfdx, state.tmp)
		end
    end
    config.numUpdates = config.numUpdates +1
  

    -- return x*, f(x) before optimization
    return x, {fx}
end


function adam(opfunc, x, config, state)
    --print('ADAM')
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
	    -- Initialization
	    state.t = state.t or 0
	    -- Exponential moving average of gradient values
	    state.m = state.m or x.new(dfdx:size()):zero()
	    -- Exponential moving average of squared gradient values
	    state.v = state.v or x.new(dfdx:size()):zero()
	    -- A tmp tensor to hold the sqrt(v) + epsilon
	    state.denom = state.denom or x.new(dfdx:size()):zero()

	    state.t = state.t + 1
	    
	    -- Decay the first and second moment running average coefficient
	    state.m:mul(beta1):add(1-beta1, dfdx)
	    state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

	    state.denom:copy(state.v):sqrt():add(epsilon)

	    local biasCorrection1 = 1 - beta1^state.t
	    local biasCorrection2 = 1 - beta2^state.t
	    
		local fac = 1
		if config.numUpdates < 10 then
		    fac = 50.0
		elseif config.numUpdates < 30 then
		    fac = 5.0
		else 
		    fac = 1.0
		end
		io.write(" ", lr/fac, " ")
        local stepSize = (lr/fac) * math.sqrt(biasCorrection2)/biasCorrection1
	    -- (2) update x
	    x:addcdiv(-stepSize, state.m, state.denom)
    end
    config.numUpdates = config.numUpdates +1
    -- return x*, f(x) before optimization
    return x, {fx}
end


-- training function

function adversarial.train(dataset_HR, dataset_LR, dataset_attr)

	model_G:training()
	model_D:training()
	epoch = epoch or 0
	local N = N or ntrain
	local dataBatchSize = opt.batchSize / 2
	local time = sys.clock()
	local err_gen = 0
  
    local inputs = torch.Tensor(dataBatchSize*3, opt.geometry[1], opt.geometry[2], opt.geometry[3])
    local targets_D = torch.Tensor(dataBatchSize*3)
    local inputs_attr_D = torch.Tensor(dataBatchSize*3, nAttr)

	local inputs_HR = torch.Tensor(dataBatchSize*2, opt.geometry[1], opt.geometry[2], opt.geometry[3])
	local inputs_LR = torch.Tensor(dataBatchSize*2, opt.geometry[1], 16, 16)
	local inputs_Attr = torch.Tensor(dataBatchSize*2, nAttr)
	local targets_G = torch.Tensor(dataBatchSize*2)
	
	-- do one epoch
	print('\n<trainer> on training set:')
	print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. sgdState_D.learningRate .. ']')
	for t = 1,N,opt.batchSize do --dataBatchSize do
		IDX_Batch = reArrangeBatch(IDX[{{t, t+dataBatchSize*2-1}}], dataset_attr)
		collectgarbage()
		
		inputs_HR = dataset_HR:index(IDX_Batch):clone()
		inputs_HR = inputs_HR:index(2, torch.LongTensor{3,2,1}):cuda()
		inputs_HR:mul(2):add(-1)

		inputs_LR:copy(dataset_LR:index(1, IDX_Batch))
		inputs_Attr:copy(dataset_attr:index(1, IDX_Batch))
--		inputs_Attr:mul(2):add(-1)
--		targets_Attr:copy(dataset_attr:index(1, IDX_Batch))
	----------------------------------------------------------------------
	-- create closure to evaluate f(X) and df/dX of discriminator
		local fevalD = function(x)
		  collectgarbage()
		  if x ~= parameters_D then -- get new parameters
			parameters_D:copy(x)
		  end

		  gradParameters_D:zero() -- reset gradients

		  --  forward pass
		  local outputs = model_D:forward({inputs, inputs_attr_D})
		  
		  if sgdState_D.trade_off then
			err_R = criterion_D:forward(outputs:narrow(1, 1, dataBatchSize), targets_D:narrow(1, 1, dataBatchSize))
			err_F = criterion_D:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, dataBatchSize*2), targets_D:narrow(1, (opt.batchSize / 2) + 1, dataBatchSize*2 ))

			local margin = opt.margin -- org = 0.3
			sgdState_D.optimize = true
			sgdState_G.optimize = true      
			if err_F < margin or err_R < margin then
			 sgdState_D.optimize = false
			end
			if err_F > (1.0-margin) or err_R > (1.0-margin) then
			 sgdState_G.optimize = false
			end
			if sgdState_G.optimize == false and sgdState_D.optimize == false then
			 sgdState_G.optimize = true 
			 sgdState_D.optimize = true
			end
			
		  end
		
		  --io.write("v1_ytc| R:", err_R,"  F:", err_F, "  ")
		  local f1 = criterion_D:forward(outputs, targets_D)

		  -- backward pass 
		  local df1_do = criterion_D:backward(outputs, targets_D)
		  model_D:backward({inputs, inputs_attr_D}, df1_do)

		  -- penalties (L1 and L2):
		  if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
			local norm,sign= torch.norm,torch.sign
			-- Loss:
			f = f + opt.coefL1 * norm(parameters_D,1)
			f = f + opt.coefL2 * norm(parameters_D,2)^2/2
			-- Gradients:
			gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
		  end

		  --print('grad D', gradParameters_D:norm())
		  return f,gradParameters_D
		end

		----------------------------------------------------------------------
		-- create closure to evaluate f(X) and df/dX of generator 
		local fevalG = function(x)
		  collectgarbage()
		  if x ~= parameters_G then -- get new parameters
			parameters_G:copy(x)
		  end
		  
		  gradParameters_G:zero() -- reset gradients

		  -- forward pass
		  local samples = model_G:forward({inputs_LR, inputs_Attr})
		  local g       = criterion_G:forward(samples, inputs_HR) 
		  
		  local samples_percep = preprocess_image(samples:clone())
	      local inputs_HR_percep = preprocess_image(inputs_HR:clone())   
		  local g_percp = PerceptionLoss:forward(samples_percep, inputs_HR_percep)  
		  err_gen       = err_gen + g
		  
		  local outputs = model_D:forward({samples, inputs_Attr})
		  local f1      = criterion_D:forward(outputs, targets_G)

		  --  backward pass
		  local df1_samples = criterion_D:backward(outputs, targets_G)
		  model_D:backward({samples, inputs_Attr}, df1_samples)
			
		  local df_G_samples = criterion_G:backward(samples, inputs_HR)   ---added by xin
		  local df_vgg = PerceptionLoss:backward(samples_percep, inputs_HR_percep):div(127.5)   
		  local df_do = model_D.modules[1].modules[1].modules[1].gradInput * opt.lambda + df_G_samples + df_vgg * opt.eta
		  model_G:backward({inputs_LR, inputs_Attr}, df_do)

		--      print('gradParameters_G', gradParameters_G:norm())
		  return f,gradParameters_G
		end

	----------------------------------------------------------------------
	-- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
	-- Get half a minibatch of real, half fake
				
		-- (1.1) Real data
		local Part1 = dataBatchSize
		local Part2 = dataBatchSize*2
		local Part3 = dataBatchSize*3
 
		inputs[{{1,Part1}}] = inputs_HR[{{1,Part1}}]:clone()
		inputs_attr_D[{{1, Part1}}] = inputs_Attr[{{1,Part1}}]:clone()
			
		-- generate images: one quater of minibatch 
		local samples_LR = inputs_LR[{{1,Part1}}]
		local samples_Attr = inputs_Attr[{{1,Part1}}]
		samples = model_G:forward({samples_LR:cuda(), samples_Attr:cuda()})
		inputs[{{Part1+1, Part2}}] = samples:clone()
		inputs_attr_D[{{Part1+1, Part2}}] = inputs_Attr[{{1,Part1}}]:clone()
		
		inputs[{{Part2+1, Part3}}] = inputs_HR[{{1, Part1}}]:clone()
		inputs_attr_D[{{Part2+1, Part3}}] = randomChange_Attr(inputs_Attr[{{1, Part1}}]):clone()

		targets_D[{{1,dataBatchSize}}]:fill(1)
		targets_D[{{dataBatchSize+1, dataBatchSize*3}}]:fill(0)

		rmsprop(fevalD, parameters_D, sgdState_D)

	----------------------------------------------------------------------
	-- (2) Update G network: maximize log(D(G(z)))
	-- noise_inputs:normal(0, 1)
	
		
		targets_G:fill(1)
		rmsprop(fevalG, parameters_G, sgdState_G)
		
		-- display progress
		nIter = nIter + 1
		if (nIter-1) % opt.saveFreq == 0 then
			netname = string.format('adversarial_net_%s',nIter)
			local filename = paths.concat(opt.save, netname)    
			os.execute('mkdir -p ' .. sys.dirname(filename))
			print('<trainer> saving network to '..filename)
--			model_D:clearState()
			model_G:clearState() 
--			for i, module in ipairs(model_G:listModules()) do
--				module:clearState()
--			end
			torch.save(filename, {G = model_G})
			
			--local to_plot = getSamples(valData_LR, dataset_attr, 100)
			local to_plot = getSamples_compare_attrflip(valData_LR, dataset_HR, dataset_attr, 50)
			torch.setdefaulttensortype('torch.FloatTensor')

			local formatted = image.toDisplayTensor({input = to_plot, nrow = 15})
			formatted:float()
			formatted = formatted:index(1,torch.LongTensor{3,2,1})

			image.save(opt.save .. '/UR_example_' .. (nIter or 0) .. '.png', formatted)
			torch.setdefaulttensortype('torch.CudaTensor')
		end
		
		-- display progress
		xlua.progress(t, ntrain)
	end -- end for loop over dataset

    -- time taken
    time = sys.clock() - time
    time = time / ntrain
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    --print(confusion)
    trainLogger:add{['MSE accuarcy1'] = err_gen/(ntrain/opt.batchSize)}

    -- next epoch
    epoch = epoch + 1

end

return adversarial