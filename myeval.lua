-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
-- Forward for batchsize data
function forward(s,e)
   -- grab a batch
   local batch_size = e-s+1
   local qinds = torch.LongTensor(batch_size):fill(0)
   local iminds = torch.LongTensor(batch_size):fill(0)
   local fv_im = torch.Tensor(batch_size,2048,14,14)
   local fr_im = torch.Tensor(batch_size,19,4097):zero()

   for i = 1, batch_size do
      qinds[i] = s+i-1
      iminds[i] = dataset['val_img_list'][qinds[i]]
      fv_im[i]:copy(h5_file_ms:read(paths.basename(val_list[iminds[i]])):all())
      fr_im[i]:copy(h5_file_frms:read(paths.basename(val_list[iminds[i]])):all())
   end   
   local fv_sorted_q = dataset['val_question']:index(1,qinds) 
   local val_labels = dataset['val_answers']:index(1,qinds)

   -- ship to gpu
   if opt.gpuid >= 0 then
      fv_sorted_q = fv_sorted_q:cuda() 
      fv_im = fv_im:cuda()
      fr_im = fr_im:cuda()
      val_labels = val_labels:cuda()
   end
   model:cuda()

   -- forward
   local scores = model:forward({fv_sorted_q, fv_im, fr_im})
   local loss = criterion:forward(scores, val_labels)
   return scores:double(), loss, val_labels
end


-- Do Prediction for validation
function validation()
   model:evaluate() -- Takes 0 second
   
   local loss_sum = 0
   local loss_evals = 0
   local right_sum = 0
   local batch_size = opt.val_batch_size
   
   local timer3 = torch.Timer()
   for i = 1, val_nqs, batch_size do
      xlua.progress(i, val_nqs); if batch_size>val_nqs-i then xlua.progress(val_nqs, val_nqs) end
      local r = math.min(i+batch_size-1, val_nqs)
      local val_scores, val_loss, val_labels = forward(i,r)
      local tmp, val_pred = torch.max(val_scores,2)
      for i = 1, val_pred:size()[1] do
         if val_pred[i][1] == val_labels[i] then
            right_sum = right_sum + 1
         end
      end
      loss_sum = loss_sum + val_loss
      collectgarbage()
   end
   loss_evals = val_nqs/batch_size
   print(string.format('It takes %.1f seconds for validation process.', timer3:time().real)) --[PAN]
   
   model:training() 
   -- collectgarbage()
   return loss_sum/loss_evals, right_sum/val_nqs
end
