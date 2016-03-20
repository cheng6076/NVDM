local Loader = {}
local stringx = require('pl.stringx')
Loader.__index = Loader

function Loader.create(data_dir)
    local self = {}
    setmetatable(self, Loader)
    local train_files = path.join(data_dir, 'train.txt')
    local test_files = path.join(data_dir, 'test.txt')
    local input_files = {train_files, test_files}

    -- construct a tensor with all the data
    local output, idx2word, word2idx = Loader.text_to_tensor(input_files)
    self.idx2word = idx2word
    self.word2idx = word2idx
    print(string.format('Word vocab size: %d', #self.idx2word))
    self.doc = output
    self.doc_size = {#(output[1]), #(output[2])}
    self.doc_idx = {0, 0}
    collectgarbage()
    return self
end

function Loader:reset_doc_pointer(split_idx, doc_idx)
    doc_idx = doc_idx or 0
    self.doc_idx[split_idx] = doc_idx
end

function Loader:next_doc(split_idx)
    self.doc_idx[split_idx] = self.doc_idx[split_idx] + 1
    if self.doc_idx[split_idx] > self.doc_size[split_idx] then
        self.doc_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.doc_idx[split_idx]
    return self.doc[split_idx][idx]
end

function Loader.text_to_tensor(input_files, max_sentence_l, input_w2v)
    print('Processing text into tensors...')
    local f
    local dindex
    local vocab_count = {} -- vocab count 
    local idx2word = {} 
    local word2idx = {}
    local output = {}  -- a table that stores word index for each document
    output[1] = {}
    output[2] = {}
  
    for split = 1,2 do
       f = io.open(input_files[split], 'r')      
       for line in f:lines() do
           dindex = {} 
           local tuple = stringx.split(line, '\t')
           local class, data = tuple[1], tuple[2]
           for rword in data:gmatch'([^%s]+)' do
               if word2idx[rword]==nil then
                   idx2word[#idx2word + 1] = rword 
                   word2idx[rword] = #idx2word
               end
               table.insert(dindex, word2idx[rword])
           end
           table.insert(output[split], torch.Tensor(dindex))
       end
       f:close()
    end
    return output, idx2word, word2idx
end

return Loader

