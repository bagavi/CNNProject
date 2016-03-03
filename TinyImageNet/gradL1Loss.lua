local GradL1Loss, parent = torch.class('nn.TVLoss', 'nn.Module')

function GradL1Loss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.l_diff = torch.Tensor()
  self.r_diff = torch.Tensor()
  self.u_diff = torch.Tensor()
  self.b_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
	self.l_diff = input:clone()
	self.l_diff[{{},{},{},{-1}}]  = 0
	self.l_diff[{{},{},{},{1,-2}}]:add(-1, input[{{},{},{},{2,-1}}] );

	self.r_diff = input:clone()
	self.r_diff[{{},{},{},{1}}] = 0
	self.r_diff[{{},{},{},{2,-1}}]:add(-1, input[{{},{},{},{1,-2}}] );


	self.b_diff = input:clone()
	self.b_diff[{{},{},{1},{}}] = 0
	self.b_diff[{{},{},{2,-1},{}}]:add(-1, input[{{},{},{1,-2},{}}] );


	self.u_diff = input:clone()
	self.u_diff[{{},{},{-1},{}}]  = 0
	self.u_diff[{{},{},{1,-2},{}}]:add(-1, input[{{},{},{2,-1},{}}] );

	-- Using pre-existing L1 class.
	L1node = nn.L1Cost()
	self.loss = L1node:forward(self.r_diff) + L1node:forward(self.l_diff) + L1node:forward(self.b_diff) + L1node:forward(self.u_diff)
	self.loss:mul(self.strength)

	-- Normalizing the loss
	self.loss:div(input:size(1)*input:size(2)*input:size(3)*input:size(4))

	return self.loss
end

function TVLoss:updateGradInput(input, gradOutput)
	-- Using pre-existing L1 class.
	L1node = nn.L1Cost()

	self.gradInput = L1node:backward(self.r_diff):clone() + L1node:backward(self.l_diff):clone() + L1node:backward(self.b_diff):clone() + L1node:backward(self.u_diff):clone()

	self.gradInput:mul(self.strength)
	self.gradInput:div(4)
	
	return self.gradInput
end
