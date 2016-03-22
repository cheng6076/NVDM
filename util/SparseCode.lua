local SparseCode, parent = torch.class('nn.SparseCode', 'nn.Module')

function SparseCode:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
end

function SparseCode:updateOutput(input)
  self.output:resize(self.outputSize):zero():float()
  local longInput = input:long()
  for i=1, input:size(1) do
    self.output[input[i]] = self.output[input[i]] + 1
  end
  return self.output
end
