netdef = {}

function netdef.ATT7b(dropout)
   -- input: {fv_sorted_q, fv_im, fr_im}
   local p = dropout or 0.5

   local model = nn.Sequential()

   :add(nn.ParallelTable()
      :add(nn.Sequential()
         :add(embedding_net_q)
         :add(encoder_net_q))  -- [GRU]
      :add(nn.Identity())
      :add(nn.Identity())
   ) -- {Q, V1, V2}  {2400, 2048x14x14, 19x4097}

   :add(nn.ConcatTable()
      :add(nn.Sequential()
         :add(nn.ConcatTable()
            :add(nn.SelectTable(2)) -- V1
            :add(nn.SelectTable(1)) -- Q
            :add(nn.SelectTable(3))) -- V2
         :add(mfa_net1)) -- [MFA1] {V1,Q,V2} -> {f1}
      
      :add(nn.Sequential()
         :add(nn.ConcatTable()
            :add(nn.SelectTable(2)) -- V1
            :add(nn.SelectTable(1)) -- Q
            :add(nn.SelectTable(3))) -- V2
         :add(mfa_net2)) -- [MFA2] {V1,Q,V2} -> {f2}
   ) -- {f1, f2}

   :add(fusion_net) -- [FUS]

   return model
end
