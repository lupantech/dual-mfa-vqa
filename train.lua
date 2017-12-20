require 'nn'
require 'rnn'
require 'dp'
require 'torch'
require 'optim'
require 'cutorch'
require 'cunn'
require 'hdf5'
require 'misc.myutils'
cjson = require('cjson') 

-------------------------------------------------------------------------------
-- [1] Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-phase',2,'training phase, 1: train on Train, 2: train on Train+Val')
cmd:option('-input_img_h5','../VQA/Features/img_features_res152_train-val.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data_train-val_test-dev_2k/vqa_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data_train-val_test-dev_2k/vqa_data_prepro.json','path to the json file containing additional info and vocab')
cmd:option('-fr_ms_h5', '../VQA/Features/faster-rcnn_features_19_train-val.h5', 'path to mscoco image faster-rcnn h5 file')
cmd:option('-input_skip','skipthoughts_model/','path to skipthoughts_params')
cmd:option('-input_seconds', 'data_train-val_test-dev_2k/seconds.json', 'path to sencond answers')
cmd:option('-vqa_type', 'vqa', 'vqa or coco-qa')

-- Loading image data to memory
cmd:option('-memory_ms', false, 'load image resnet feature to memory')
cmd:option('-memory_frms', false, 'load image fast-rcnn feature to memory')

-- Validation settings
cmd:option('-val',false,'running validation')
cmd:option('-val_nqs', -1,'# of validation questions')
cmd:option('-val_batch_size',64,'batch_size for each validation iteration')

-- Basic model parameter settings
cmd:option('-batch_size',100,'batch_size for each iterations')
cmd:option('-input_encoding_size', 620, 'the encoding size of each token in the vocabulary')
cmd:option('-rnn_size',2400,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-common_embedding_size', 1200, 'size of the common embedding vector')
cmd:option('-num_output', 2000, 'number of output answers')
cmd:option('-dropout', .5, 'dropout probability for joint functions')
cmd:option('-glimpse', 2, '# of glimpses')
cmd:option('-clipping', 10, 'gradient clipping')
cmd:option('-seconds', true, 'usage of second candidate answers')
cmd:option('-best_acc', 55, 'best test accuracy by now')

-- Overall attetion model settings
cmd:option('-run_id','207','running  model id') -- eg., #10
cmd:option('-MFA_model_name', 'MFA1g', 'MFA1 model name')
cmd:option('-MFA2_model_name', 'MFA2g', 'MFA2 model name')
cmd:option('-fusion_model_name', 'ADD', 'fusion model name')
cmd:option('-att_model_name', 'ATT7b', 'attention model name')
cmd:option('-model_label', 'dual-mfa', 'model label name') -- eg., dual-mfa

-- Optimizer parameter settings
cmd:option('-learning_rate',3e-4,'learning rate for rmsprop')
cmd:option('-learning_rate_decay_start', 0, 'at which iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 100, 'how many iterations to drop LR by 0.1?')
cmd:option('-max_iters', 350000, 'max # of iterations to run for ')
cmd:option('-optimizer','rmsprop','opimizer')
cmd:option('-winit_method', '', 'initial model paprameters')

-- Check point and quick check
cmd:option('-save_checkpoint_every', 10000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'model/', 'folder to save checkpoints')
cmd:option('-load_checkpoint_path', '', 'path to saved checkpoint')
cmd:option('-previous_iters', 0, 'previous # of iterations to get previous learning rate')
cmd:option('-kick_interval', 50000, 'interval of kicking the learning rate as its double')
cmd:option('-skip_save_model', false, 'skip saving t7 model')
cmd:option('-losses_log_every', 100, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')
cmd:option('-cg_every', 10, 'How often do we collectgarbage in the training process')
cmd:option('-quick_check', false, 'quick check for code')
cmd:option('-quickquick_check', false, 'very quick check for code')

-- Misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 1231, 'random number generator seed to use')
cmd:option('-nGPU', 1, 'how many gpu to use. 1 = use 1 GPU')

opt = cmd:parse(arg)

---------------------------
-- Update Parameters
---------------------------
if opt.quick_check then -- quick check for code
   opt.memory_ms = false
   opt.memory_frms = false
   opt.skip_save_model = true -- don't save model
   opt.losses_log_every = 100
   opt.save_checkpoint_every = 300 
end

if opt.quickquick_check then -- veryquick check for code
   opt.memory_ms = false
   opt.memory_frms = false
   opt.skip_save_model = true -- don't save model
   opt.losses_log_every = 10
   opt.save_checkpoint_every = 30
end

opt.iterPerEpoch = 360000 / opt.batch_size
-- opt.kick_interval = math.floor(opt.kick_interval/(opt.learning_rate/3e-4))
-- opt.learning_rate = math.sqrt(opt.batch_size/100) * opt.learning_rate

-- Training stage
if opt.phase == 1 then -- 1: train on Train, 2: train on Train+Val
   opt.val = true
   opt.input_ques_h5 = 'data_train_test-dev_2k/vqa_data_prepro.h5'
   opt.input_json = 'data_train_test-dev_2k/vqa_data_prepro.json'
   opt.iterPerEpoch = 240000 / opt.batch_size
end

-- VQA dataset type
if opt.vqa_type == 'coco-qa' then
   opt.input_img_h5 = '../VQA/Features/coco_img_features_res152.h5'
   opt.input_ques_h5 = 'data_coco/cocoqa_data_prepro.h5'
   opt.input_json = 'data_coco/cocoqa_data_prepro.json'
   opt.seconds = false
   opt.num_output = 430
   opt.phase = 2
   opt.val = true -- for coco-qa, Val dataset = Test dataset
   opt.val_nqs = -1
   opt.kick_interval = 1000000
   opt.best_acc = 60
   -- opt.learning_rate = 4e-4
end

print(opt)


------------------------------------------------------------------------
-- [2] Setting the parameters
------------------------------------------------------------------------
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

require 'misc.RNNUtils'

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1)
end

local model_name = opt.vqa_type..'_model' -- vqa_model or coco-qa_model
if opt.run_id ~= '' then model_name = model_name..'_#'..opt.run_id end
model_name = model_name..'_'..opt.MFA_model_name..'_'..opt.MFA2_model_name..'_'..opt.fusion_model_name..'_'..opt.att_model_name
model_name = model_name..'_'..opt.model_label
print('output model name is: ', model_name)
-- 'vqa_model_#207_MFA1g_MFA2g_ADD_ATT7b_dualmfa.t7'

local model_path = opt.checkpoint_path
local batch_size = opt.batch_size
local embedding_size_q = opt.input_encoding_size
local rnn_size_q = opt.rnn_size
local common_embedding_size = opt.common_embedding_size
local noutput = opt.num_output
local dropout = opt.dropout
local glimpse = opt.glimpse
local decay_factor = 0.99997592083
-- local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
local question_max_length = 26
local seconds = readAll(opt.input_seconds)
paths.mkdir(model_path)


------------------------------------------------------------------------
-- [3] Loading Dataset
------------------------------------------------------------------------
-- DataLoader: loading input_json json file
print('Loading input_json json file: ', opt.input_json)
local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local count = 0
for i, w in pairs(json_file['ix_to_word']) do count = count + 1 end
local vocabulary_size_q = count

-- DataLoader: loading input_ques h5 file
print('Loading input_ques h5 file: ', opt.input_ques_h5)
dataset = {}
local h5_file = hdf5.open(opt.input_ques_h5, 'r')
local nhimage = 2048

dataset['question'] = h5_file:read('/ques_train'):all()
dataset['question_id'] = h5_file:read('/ques_id_train'):all()
dataset['lengths_q'] = h5_file:read('/ques_len_train'):all()
dataset['img_list'] = h5_file:read('/img_pos_train'):all()
dataset['answers'] = h5_file:read('/answers'):all()
print('Train Dataset size:', dataset['question_id']:size()[1])

-- DataLoader: loading validation data
if opt.val then
   require 'myeval'
   local total_val_nqs = h5_file:read('/ques_val'):all():size(1)
   if opt.val_nqs == -1 then 
      val_nqs = total_val_nqs
   else 
      val_nqs = math.min(total_val_nqs, opt.val_nqs)
   end
   local val_qinds = torch.randperm(total_val_nqs):sub(1,val_nqs):long()
   val_qinds = torch.LongTensor(val_qinds)
   dataset['val_question'] = h5_file:read('/ques_val'):all():index(1,val_qinds)
   dataset['val_lengths_q'] = h5_file:read('/ques_len_val'):all():index(1,val_qinds)
   dataset['val_img_list'] = h5_file:read('/img_pos_val'):all():index(1,val_qinds)
   dataset['val_ques_id'] = h5_file:read('/ques_id_val'):all():index(1,val_qinds)
   dataset['val_answers'] = h5_file:read('/ans_val'):all():index(1,val_qinds)
end
h5_file:close()

-- Train image list 
local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
   table.insert(train_list, imname)
end
dataset['question'] = right_align(dataset['question'],dataset['lengths_q'])

-- Val image list
if opt.val then
   val_list={}
   for i,imname in pairs(json_file['unique_img_val']) do
      table.insert(val_list, imname) -- {"val2014/COCO_val2014_000000350623.jpg"}
   end
   dataset['val_question'] = right_align(dataset['val_question'],dataset['val_lengths_q'])
end

-- DataLoader: loading image h5 file: h5_file_ms
print('Loading ms image h5 file: ', opt.input_img_h5)
h5_file_ms = hdf5.open(opt.input_img_h5, 'r') -- from Disk
if opt.memory_ms then -- from Memory
   print('Loading h5_file_ms_data to memory...')
   local unique_img_train_num = #json_file['unique_img_train'] -- 82627
   h5_file_ms_data = torch.Tensor(unique_img_train_num,2048,14,14) -- 14.95G RAM Memory for 10k images, torch.FloatTensor
   for imgind = 1, unique_img_train_num do
      xlua.progress(imgind, unique_img_train_num); if imgind == unique_img_train_num then xlua.progress(imgind, imgind) end
      local image_name = paths.basename(json_file['unique_img_train'][imgind])
      h5_file_ms_data[imgind]:copy( h5_file_ms:read(image_name):all() )
   end
   print('Done!')
end

-- DataLoader: loading fr_ms image h5 ile: h5_file_frms
print('Loading fr_ms image h5 file: ', opt.fr_ms_h5)
h5_file_frms = hdf5.open(opt.fr_ms_h5, 'r') -- from Disk
if opt.memory_frms then -- from Memory
   print('Loading h5_file_frms_data to memory...')
   local unique_img_train_num = #json_file['unique_img_train'] -- 82627
   h5_file_frms_data = torch.Tensor(unique_img_train_num,19,4097)
   for imgind = 1, unique_img_train_num do
      xlua.progress(imgind, unique_img_train_num); if imgind == unique_img_train_num then xlua.progress(imgind, imgind) end
      local image_name = paths.basename(json_file['unique_img_train'][imgind])
      h5_file_frms_data[imgind]:copy( h5_file_frms:read(image_name):all() )
   end
   print('Done!')
end

collectgarbage() 


------------------------------------------------------------------------
-- [4] Design Parameters and Network Definitions
------------------------------------------------------------------------
print('Building the model...')

----------------------------------
-- GRU Model
----------------------------------
-- question embedding model: GRU
if opt.vqa_type == 'vqa' then
   -- skip-thought vectors
   -- lookup = nn.LookupTableMaskZero(vocabulary_size_q, embedding_size_q)
   if opt.num_output == 1000 then lookupfile = 'lookup_fix.t7'
   elseif opt.num_output == 2000 then lookupfile = 'lookup_2k.t7' 
   elseif opt.num_output == 3000 then lookupfile = 'lookup_3k.t7' 
   end
   lookup = torch.load(paths.concat(opt.input_skip, lookupfile))
   assert(lookup.weight:size(1) == vocabulary_size_q+1)  -- +1 for zero
   assert(lookup.weight:size(2) == embedding_size_q)
   gru = torch.load(paths.concat(opt.input_skip, 'gru.t7'))
   -- Bayesian GRUs have right dropouts
   rnn_model = nn.GRU(embedding_size_q, rnn_size_q, false, .25, true)
   skip_params = gru:parameters()
   rnn_model:migrate(skip_params)

elseif opt.vqa_type == 'coco-qa' then
   lookup = nn.LookupTableMaskZero(vocabulary_size_q, embedding_size_q)
   rnn_model = nn.GRU(embedding_size_q, rnn_size_q, false, .25, true)
end

rnn_model:trimZero(1)
gru = nil

-- encoder: RNN body
encoder_net_q = nn.Sequential()
            :add(nn.Sequencer(rnn_model))
            :add(nn.SelectTable(question_max_length))

-- embedding: word-embedding
embedding_net_q = nn.Sequential()
            :add(lookup)
            :add(nn.SplitTable(2))

collectgarbage()

----------------------------------
-- Dual-MFA
----------------------------------
-- MFA1
print('load MFA1 model:', opt.MFA_model_name)
require('netdef.MFA')
mfa_net1 = netdef[opt.MFA_model_name](rnn_size_q,nhimage,common_embedding_size,glimpse) -- 1200x2 --> 2000
mfa_net1:getParameters():uniform(-0.08, 0.08) 

-- MFA2
print('load MFA2 model:', opt.MFA2_model_name)
require('netdef.MFA')
mfa_net2 = netdef[opt.MFA2_model_name](rnn_size_q,4097,common_embedding_size,glimpse) -- 1200x2 --> 2000
mfa_net2:getParameters():uniform(-0.08, 0.08) 

-- FUS
require('netdef.FUS')
fusion_net = netdef[opt.fusion_model_name](common_embedding_size, noutput, glimpse) -- 1200x2 --> 2000

-- Oral attention model
require('netdef.ATT')
model = netdef[opt.att_model_name]()

-- Criterion
criterion = nn.CrossEntropyCriterion()

-- print(model)
collectgarbage()

----------------------------------
-- GPU and Multi-GPU
----------------------------------
if opt.gpuid >= 0 then
   print('shipped data function to cuda...')
   model = model:cuda()
   criterion = criterion:cuda()
   cudnn.fastest = true
   cudnn.benchmark = true
end

-- Multi-GPU
if opt.nGPU > 1 then
   local gpus = torch.range(1, opt.nGPU):totable()
   local fastest, benchmark = cudnn.fastest, cudnn.benchmark
   print('nGPU...')

   local dpt = nn.DataParallelTable(1, true, true)
      :add(model, gpus)
      :threads(function()
         local cudnn = require 'cudnn'
         local rnn = require 'rnn' -- neccessary for rnn model
         local nngraph = require 'nngraph'  -- wyang: to work with nngraph on multi-GPUs
                                            -- https://github.com/torch/cunn/issues/241
         cudnn.fastest, cudnn.benchmark = fastest, benchmark
      end)
   dpt.gradInput = nil
   model = dpt:cuda()
end

----------------------------------
-- Optimization Setting
----------------------------------
if opt.winit_method == 'xavier' then
   --https://github.com/e-lab/torch-toolbox/blob/master/Weight-init/README.md
   print('initial: xavier' )
   local method = 'xavier'
   model = require('misc.weight-init')(model, method)
end

w,dw = model:getParameters()

if paths.filep(opt.load_checkpoint_path) then
   print('loading checkpoint model...')
   model_param = torch.load(opt.load_checkpoint_path); -- load the model
   w:copy(model_param) -- use the precedding parameters
end

-- optimization parameter
local optimize = {} 
optimize.maxIter = opt.max_iters
optimize.learningRate = opt.learning_rate
optimize.update_grad_per_n_batches=1

optimize.winit = w
print('nParams =', optimize.winit:size(1))
print('decay_factor =', decay_factor)

collectgarbage()


------------------------------------------------------------------------
-- [5] Next batch for training
------------------------------------------------------------------------
function dataset:next_batch(batch_size)
   local qinds = torch.LongTensor(batch_size):fill(0) 
   local iminds = torch.LongTensor(batch_size):fill(0)  
   local nqs = dataset['question']:size(1)
   local fv_im = torch.Tensor(batch_size,2048,14,14)
   local fr_im = torch.Tensor(batch_size,19,4097):zero()

   for i = 1,batch_size do 
      qinds[i] = torch.random(nqs) -- question_id, e.g, 9947
      iminds[i] = dataset['img_list'][qinds[i]] -- image index in unique img list, e.g, 73550/82627
   end

   if opt.memory_ms then -- Memory
      fv_im = h5_file_ms_data:index(1,iminds) -- 0.02x second for 100 data
   else 
      for i = 1,batch_size do
         fv_im[i]:copy( h5_file_ms:read(paths.basename(train_list[iminds[i]])):all() )
      end
   end

   if opt.memory_frms then -- Memory
      fr_im = h5_file_frms_data:index(1,iminds)
   else 
      for i = 1,batch_size do
         fr_im[i]:copy( h5_file_frms:read(paths.basename(train_list[iminds[i]])):all() )
      end
   end

   local fv_sorted_q = dataset['question']:index(1,qinds) 
   local labels = dataset['answers']:index(1,qinds)

   -- using second candidate answer sampling
   if opt.seconds then
      local sampling = torch.rand(batch_size)
      local qids = dataset['question_id']:index(1,qinds) 
      for i = 1,batch_size do
         local second = seconds[tostring(qids[i])]
         if second then
            -- print('seconds hit!')
            if sampling[i]<second.p then
               -- print('seconds sampled! p=', second.p)
               -- print(json_file.ix_to_ans[tostring(labels[i])]..'=>'..
                     -- json_file.ix_to_ans[tostring(second.answer)])
               labels[i] = second.answer
            end
         end
      end
   end
   
   -- ship to gpu
   if opt.gpuid >= 0 then
      fv_sorted_q=fv_sorted_q:cuda() 
      fv_im = fv_im:cuda()
      fr_im = fr_im:cuda()
      labels = labels:cuda()
   end
   -- print('fv_sorted_q=',fv_sorted_q:size(), 'fv_im=',fv_im:size(), 'fr_im=',fr_im:size())
   return fv_sorted_q,fv_im,fr_im,labels

end


------------------------------------------------------------------------
-- [6] Objective Function and Optimization
------------------------------------------------------------------------
function JdJ(x)
   -- clear gradients
   dw:zero()

   -- grab a batch
   local timer_data = torch.Timer()
   local fv_sorted_q, fv_im, fr_im, labels
   fv_sorted_q, fv_im, fr_im, labels = dataset:next_batch(batch_size) --[PAN]
   -- print ("fv_sorted_q",fv_sorted_q:size(),fv_sorted_q)
   time_data_100 = time_data_100 + timer_data:time().real

   local timer_gpu = torch.Timer()
   local scores = model:forward({fv_sorted_q, fv_im, fr_im}) --[PAN]
   local f = criterion:forward(scores, labels)
   local dscores = criterion:backward(scores, labels)
   model:backward(fv_sorted_q, dscores)
   time_gpu_100 = time_gpu_100 + timer_gpu:time().real

   gradients = dw
   if opt.clipping > 0 then gradients:clamp(-opt.clipping, opt.clipping) end
   if running_avg == nil then
      running_avg = f
   end
   running_avg = running_avg*0.95 + f*0.05

   return f, gradients
end


------------------------------------------------------------------------
-- [7] Training
------------------------------------------------------------------------
-- get previous learning rate when using previous checkpoint model
optimize.learningRate = optimize.learningRate*decay_factor^opt.previous_iters
optimize.learningRate = optimize.learningRate*2^math.min(2, math.floor(opt.previous_iters/opt.kick_interval))
local state = {}
paths.mkdir(model_path..'save')

timer = torch.Timer()
timer_100_iter = torch.Timer()
time_data_100 = 0.0
time_gpu_100 = 0.0

collectgarbage()

for iter = opt.previous_iters + 1, opt.max_iters do
   -- forward and backward pass, learningRate update
   if iter == opt.kick_interval or iter == opt.kick_interval*2 then -- double learning rate at two iteration points
      optimize.learningRate = optimize.learningRate*2
      print('learining rate:', optimize.learningRate)
   end
   if opt.previous_iters == iter-1 then
      print('learining rate:', optimize.learningRate)
   end
   optim[opt.optimizer](JdJ, optimize.winit, optimize, state)
   if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
      optimize.learningRate = optimize.learningRate * decay_factor -- set the decayed rate
   end 

   -- printing training loss and time
   if iter <= 100 or iter%opt.losses_log_every == 0 then -- 100, small value like 1 for initial watch
      local loss_str = string.format('training loss: %.5f, on iter: %d/%d.  ', running_avg, iter, opt.max_iters, opt.max_iters)
      local time_str = string.format('Total/Data/GPU time: %d/%d/%.1f s.', timer_100_iter:time().real, time_data_100, time_gpu_100)
      print(loss_str .. time_str)
      timer_100_iter:reset(); time_data_100 = 0.0; time_gpu_100 = 0.0;
   end

   -- saving t7 model eg., every 10000 iters
   if iter%opt.save_checkpoint_every == 0 then
      if opt.val then
         collectgarbage()
         val_loss, val_accu = validation()
         collectgarbage()
         print(string.format('validation loss = %.5f, accuracy for %d samples = %.2f', val_loss,val_nqs,100*val_accu))
         if 100*val_accu > opt.best_acc then  -- have validation
            opt.best_acc = 100*val_accu
            print(string.format('[Best Acc]: %.2f', 100*val_accu))
            if not opt.skip_save_model then -- save model
               torch.save(string.format(model_path..'save/'..model_name..'_iter%d.t7',iter),w) 
               print(string.format('Saving model: '..model_name..'_iter%d.t7',iter))
            end
         end
      end
      if not opt.val and not opt.skip_save_model then -- phase 2 for 'vqa' dataset  
         torch.save(string.format(model_path..'save/'..model_name..'_iter%d.t7',iter),w) 
         print(string.format('Saving model: '..model_name..'_iter%d.t7',iter))
      end
      local hours = timer:time().real/3600
      local minutes = (timer:time().real%3600)/60
      print(string.format('It takes %d hours %.1f minites for %d iters.\n', hours, minutes, iter))
   end

   if iter%opt.cg_every == 0 then -- change this to smaller value if out of the memory
      collectgarbage() -- takes 0.2 second
   end
end

-- Saving the final model
print('Training DONE!')
torch.save(string.format(model_path..model_name..'.t7',i),w) 
h5_file_ms:close()
h5_file_frms:close()
print('Closing H5 File DONE!')
