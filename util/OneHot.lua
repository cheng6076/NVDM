local OneHot, parent = torch.class('nn.OneHot', 'nn.Module')

function OneHot:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
end

function OneHot:updateOutput(input)
  self.output:resize(input:size(1), self.outputSize):zero()
  local longInput = input:long()
  self.output:scatter(2, longInput:view(-1, 1), 1)
  return self.output
end

