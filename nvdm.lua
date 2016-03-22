require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'util.SparseCode'
require 'util.KLDCriterion'
require 'util.Sampler'
model_utils = require 'util.model_utils'
Loader = require 'util.Loader'
require 'util.misc'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-data_dir', '/afs/inf.ed.ac.uk/user/s15/s1537177/20news/', 'path of the dataset')
cmd:option('-max_epochs', 50, 'number of full passes through the training data')
cmd:option('-ntopics', 50, 'dimensionality of word embeddings')
cmd:option('-batch_size', 10, 'number of words in a mini-batch')
cmd:option('-hidden', 500, 'number of words in a mini-batch')
cmd:option('-dropout',0.5,'dropout. 0 = no dropout')
cmd:option('-coefL2',0.001,'regularization constanti for l2 norm.')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every', 8000, 'save when seeing n examples')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint', 'checkpoint.t7', 'start from a checkpoint if a valid checkpoint.t7 file is given')
cmd:option('-beta1', 0.9, 'momentum parameter 1')
cmd:option('-beta2', 0.999, 'momentum parameter 2')
cmd:option('-learningRate', 0.001, 'learning rate')
cmd:option('-decayRate',0.97,'decay rate for sgd')
cmd:option('-decay_when',0.1,'decay if validation does not improve by more than this much')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-max_grad_norm',5,'normalize gradients at')
-- GPU/CPU
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1=yes). this should greatly speed up convolutions')
cmd:option('-time', 0, 'print batch times')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- load necessary packages depending on config options
if opt.gpuid >= 0 then
   print('using CUDA on GPU ' .. opt.gpuid .. '...')
   require 'cutorch'
   require 'cunn'
   cutorch.setDevice(opt.gpuid + 1)
end
if opt.cudnn == 1 then
  assert(opt.gpuid >= 0, 'GPU must be used if using cudnn')
  print('using cudnn...')
  require 'cudnn'
end

-- create data loader
loader = Loader.create(opt.data_dir)
opt.vocab_size = #loader.idx2word
local SparseCode = nn.SparseCode(opt.vocab_size)

-- model
protos = {}
protos.enc = nn.Sequential():add(nn.Linear(opt.vocab_size,opt.hidden)):add(nn.ReLU()):add(nn.Linear(opt.hidden,opt.hidden)):add(nn.ReLU()):add(nn.ConcatTable():add(nn.Linear(opt.hidden, opt.ntopics)):add(nn.Linear(opt.hidden, opt.ntopics)))
word_vec_layer = nn.Linear(opt.ntopics,opt.vocab_size)
word_vec_layer.name = 'word_vec'
protos.dec = nn.Sequential():add(word_vec_layer):add(nn.LogSoftMax())
protos.sampler = nn.Sampler()
protos.Criterion = nn.ClassNLLCriterion()
protos.KLD = nn.KLDCriterion()
-- ship to gpu
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end
-- params and grads
params, grad_params = model_utils.combine_all_parameters(protos.enc, protos.dec)
params:uniform(-0.01, 0.01)
print('number of parameters in the model: ' .. params:nElement())

-- query word vectors for visualization
function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vec' then
      word_vec = layer
    end
  end
end
protos.dec:apply(get_layer)


function eval_split()
  local n = loader.doc_size[2]
  loader:reset_doc_pointer(2)
  local perp = 0
  for i=1, n do
    local next_doc = loader:next_doc(2)
    local d = SparseCode:forward(next_doc)
    local label = next_doc
    -- load data
    if opt.gpuid >= 0 then
      d = d:float():cuda()
      label = label:cuda()
    end
    -- Forward pass
    local theta_mean, theta_var = unpack(protos.enc:forward(d))
    local kld = protos.KLD:forward(theta_mean, theta_var)
    local theta = protos.sampler:forward({theta_mean, theta_var})
    for i=1, 19 do
      theta:add(protos.sampler:forward({theta_mean, theta_var}))
    end
    theta:mul(0.05)
    local theta_all = torch.repeatTensor(theta, label:size(1), 1)
    local w_p = protos.dec:forward(theta_all)
    local w_error = protos.Criterion:forward(w_p, label)
    local upper_bond = w_error * label:size(1) + kld

    perp = perp + upper_bond / label:size(1)
  end
  perp = perp / n
  perp = torch.exp(perp)
  return perp
end

--training
function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()
  local d = next_document
  local label = next_label
  -- load data
  if opt.gpuid >= 0 then
    d = d:float():cuda()
    label = label:cuda()
  end
  -- Forward pass
  local theta_mean, theta_var = unpack(protos.enc:forward(d))
  local theta = protos.sampler:forward({theta_mean, theta_var})
  local kld_theta = protos.KLD:forward(theta_mean, theta_var)  
  local theta_all = torch.repeatTensor(theta, label:size(1), 1)
  local w_p = protos.dec:forward(theta_all)
  local w_error = protos.Criterion:forward(w_p, label)

  -- Backward pass
  local der_wp = protos.Criterion:backward(w_p, label)
  local der_theta_all = protos.dec:backward(theta_all, der_wp)
  local der_theta = der_theta_all:sum(1):squeeze()

  local der_sampler = protos.sampler:backward({theta_mean, theta_var}, der_theta)
  local der_theta_mean, der_theta_var = der_sampler[1], der_sampler[2]
  local der_kld = protos.KLD:backward(theta_mean, theta_var)
  der_theta_mean:add(der_kld[1])
  der_theta_var:add(der_kld[2])
  local der_d = protos.enc:backward(d, {der_theta_mean, der_theta_var})

  local grad_norm, shrink_factor
  grad_norm = torch.sqrt(grad_params:norm()^2)
  if grad_norm > opt.max_grad_norm then
    shrink_factor = opt.max_grad_norm / grad_norm
    grad_params:mul(shrink_factor)
  end
  grad_params:add(params:clone():mul(opt.coefL2))
  local loss = w_error + kld_theta
  return loss, grad_params
end

function nearest_words(k, topn)
  local weights = word_vec.weight[{{}, k}]:clone()
  local sorted, indices = torch.sort(weights)
  local result = {}
  for i = 1, topn do
    table.insert(result, loader.idx2word[indices[i]])
  end
  return result
end

-- start training
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learningRate, beta1 = opt.beta1, beta2 = opt.beta2}
local iterations = opt.max_epochs * loader.doc_size[1]
for i = 1, iterations do
  -- train 
  local epoch = i / loader.doc_size[1]
  local timer = torch.Timer()
  local time = timer:time().real
  local input = loader:next_doc(1)
  next_document = SparseCode:forward(input)
  next_label = input  
  local _, loss = optim.adam(feval, params, optim_state)
  train_losses[i] = loss[1] 

  if i % opt.print_every == 0 then
    print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f", i, iterations, epoch, train_losses[i]))
  end

  -- validate and save checkpoints
  if epoch == opt.max_epochs or i % opt.save_every == 0 then
    for k=1, 20 do
      local topic = nearest_words(k, 10) 
      print (topic)
    end
    local test_perp = eval_split() 
    print ("test perplexity is ", test_perp)
  end
  
  -- misc
  if i%5==0 then collectgarbage() end
  if opt.time ~= 0 then
     print("Batch Time:", timer:time().real - time)
  end
end
