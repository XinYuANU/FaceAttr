require 'nn'
require 'torch'

local MBCECriterion, parent = torch.class('nn.MBCECriterion', 'nn.Criterion')

function MBCECriterion:__init()
	self.nAttr = 18
    parent.__init(self)
	self.input = torch.Tensor()
	self.target = torch.Tensor()
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor()
	self.w = {}
	for i = 1, self.nAttr do
		table.insert(self.w, 1)
	end
	
	self.w[18] = 10  -- young
	self.w[15] = 10  -- no_beard
	
	self.weights = torch.Tensor{self.w}
	
end

function MBCECriterion:updateOutput(input, target)
	
	self.input:resizeAs(input):copy(input)
	self.target:resizeAs(target):copy(target)
	
	local input_log1 = torch.log(self.input)
	local loss1 = torch.cmul(input_log1, self.target)
	local input_log2 = torch.log(1-self.input)
	local loss2 = torch.cmul(input_log2, (1-self.target))
	local loss  = -loss1 - loss2
	--local output = torch.mean(torch.mean(loss,2),1)
	--self.output = output:squeeze()
	
	self.output:resizeAs(loss):copy(loss)

    return self.output
end

function MBCECriterion:updateGradInput(input, target)
	
	local num = input:size(1)
	local num_attr = input:size(2)
	local weights  = torch.expand(self.weights, target:size())
			
	local nominator  = input:cmul(weights)  + input:cmul(target):cmul(1-weights) - target
	local denominator = torch.cmul(input, 1-input)
	local mask = torch.lt(denominator, 1e-5)
	denominator[mask] = 1e-3
	local grad = nominator:cdiv(denominator)
	grad:div(num)

	for i = 1, num_attr do
		if self.w[i] ~= 1 then
			grad:select(2, i):clamp(-300, 300)
		else
			grad:select(2, i):clamp(-30, 30)
		end
	end
	self.gradInput:resizeAs(input):copy(grad)

--	self.gradInput:resizeAs(input):copy(nominator/num)
--	print(self.gradInput:norm())
	return self.gradInput
end