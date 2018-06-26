require 'nn'
require 'torch'

local MBCECriterion, parent = torch.class('nn.MBCECriterion', 'nn.Criterion')

function MBCECriterion:__init()
    parent.__init(self)
	self.input = torch.Tensor()
	self.target = torch.Tensor()
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor()

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
		
	local nominator  = input - target
	local denominator = torch.cmul(input, 1-input)
	local mask = torch.lt(denominator, 1e-5)
	denominator[mask] = 1e-2
	local grad = nominator:cdiv(denominator)
	grad:div(num)
	self.gradInput:resizeAs(input):copy(grad)
	self.gradInput:clamp(-10, 10)
--	self.gradInput:resizeAs(input):copy(nominator/num)
--	print(self.gradInput:norm())
	return self.gradInput
end