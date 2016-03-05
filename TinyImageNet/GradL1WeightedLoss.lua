GradL1WeightedLoss, parent = torch.class('nn.GradL1WeightedLoss', 'nn.Criterion')

function GradL1WeightedLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.l_diff = torch.Tensor()
  self.r_diff = torch.Tensor()
  self.u_diff = torch.Tensor()
  self.b_diff = torch.Tensor()
  self.loss   = 0
end

function GradL1WeightedLoss:updateOutput(input, mask)
    expfactor = -25
	self.l_diff = input:clone()
	self.l_diff[{{},{},{},{-1}}]  = 0
	self.l_diff[{{},{},{},{1,-2}}]:add(-1, input[{{},{},{},{2,-1}}] );

	self.l_mask_diff = mask:clone()
	self.l_mask_diff[{{},{},{},{-1}}]  = 0
	self.l_mask_diff[{{},{},{},{1,-2}}]:add(-1, mask[{{},{},{},{2,-1}}] );
    self.l_mask_diff:abs()
    self.l_mask_diff:mul(expfactor)
    self.l_mask_diff:exp()
   
    
	self.r_diff = input:clone()
	self.r_diff[{{},{},{},{1}}] = 0
	self.r_diff[{{},{},{},{2,-1}}]:add(-1, input[{{},{},{},{1,-2}}] );

	self.r_mask_diff = mask:clone()
	self.r_mask_diff[{{},{},{},{1}}] = 0
	self.r_mask_diff[{{},{},{},{2,-1}}]:add(-1, input[{{},{},{},{1,-2}}] );
    self.r_mask_diff:abs()
    self.r_mask_diff:mul(expfactor)
    self.r_mask_diff:exp()


	self.b_diff = input:clone()
	self.b_diff[{{},{},{1},{}}] = 0
	self.b_diff[{{},{},{2,-1},{}}]:add(-1, input[{{},{},{1,-2},{}}] );

	self.b_mask_diff = mask:clone()
	self.b_mask_diff[{{},{},{1},{}}] = 0
	self.b_mask_diff[{{},{},{2,-1},{}}]:add(-1, input[{{},{},{1,-2},{}}] );
    self.b_mask_diff:abs()
    self.b_mask_diff:mul(expfactor)
    self.b_mask_diff:exp()


	self.u_diff = input:clone()
	self.u_diff[{{},{},{-1},{}}]  = 0
	self.u_diff[{{},{},{1,-2},{}}]:add(-1, input[{{},{},{2,-1},{}}] );

	self.u_mask_diff = mask:clone()
	self.u_mask_diff[{{},{},{-1},{}}]  = 0
	self.u_mask_diff[{{},{},{1,-2},{}}]:add(-1, input[{{},{},{2,-1},{}}] );
    self.u_mask_diff:abs()
    self.u_mask_diff:mul(expfactor)
    self.u_mask_diff:exp()

	-- Using pre-existing L1 class.
	L1node = nn.L1Cost()
	self.loss = L1node:forward(self.r_diff:cmul(self.r_mask_diff)) + 
                L1node:forward(self.l_diff:cmul(self.l_mask_diff)) + 
                L1node:forward(self.b_diff:cmul(self.b_mask_diff)) + 
                L1node:forward(self.u_diff:cmul(self.u_mask_diff))
	self.loss = self.loss*self.strength
	self.loss = self.loss/4

	-- Normalizing the loss
	self.loss = self.loss/(input:size(1)*input:size(2)*input:size(3)*input:size(4))

	return self.loss
end

function GradL1WeightedLoss:updateGradInput(input, gradOutput)
	-- Using pre-existing L1 class.
	L1node = nn.L1Cost()

	self.gradInput = L1node:backward(self.r_diff):clone():cmul(self.r_mask_diff) +
                     L1node:backward(self.l_diff):clone():cmul(self.l_mask_diff) +
                     L1node:backward(self.b_diff):clone():cmul(self.b_mask_diff) +
                     L1node:backward(self.u_diff):clone():cmul(self.u_mask_diff)

	self.gradInput:mul(self.strength)
	self.gradInput:div(4)
	self.gradInput:div(input:size(1)*input:size(2)*input:size(3)*input:size(4))

    return self.gradInput
end
