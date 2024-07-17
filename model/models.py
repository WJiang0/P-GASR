import torch.nn as nn
import torch
from lib.utils import masked_mae_loss, masked_mae_loss2

from model.layers import (
    PhyEncoder,
    MLP, 
)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # physics-guided encoder
        self.PE = PhyEncoder(Kt=3, Ks=3, blocks=[[3, 1, args.d_model, int(args.d_model//2)], [3, 1, args.d_model, int(args.d_model//2)]], 
                        input_length=args.input_length, num_nodes=args.num_nodes, droprate=args.dropout)
        
        # mlp predictors
        self.mlp_pe = MLP(args.d_model, args.d_output)
       
        self.mae = masked_mae_loss(mask_value=5.0)
        self.mae2 = masked_mae_loss2(mask_value=5.0)
        self.args = args
    
    def forward(self, x, graph):
        pe_embedd = self.PE(x, graph) # x: n,l,v,c; graph: v,v 

        return pe_embedd

    def predict(self, pe_embedd):
        pe_pred = self.mlp_pe(pe_embedd)
        return pe_pred
    
    def pe_predict(self, pe_embedd):
        pe_pred = self.mlp_pe(pe_embedd)
        self.z_in = pe_pred[:,:,:,0]
        self.z_out = pe_pred[:,:,:,1]
        return pe_pred

    def loss(self, pe_embedd, y_true, scaler, mode, pe_weights=None):
        if mode == 're-train':
            loss = self.pred_loss(pe_embedd, y_true, scaler, mode, pe_weights)
        else:
            loss = self.pred_loss(pe_embedd, y_true, scaler, mode)

        return loss

    def pred_loss(self, pe_embedd, y_true, scaler, mode, pe_weights=None):
        y_pred = scaler.inverse_transform(self.predict(pe_embedd))
        y_true = scaler.inverse_transform(y_true)
        
        if mode == 're-train':
            loss1 = self.args.yita * self.mae2(y_pred[..., 0], y_true[..., 0], pe_weights) + \
                    (1 - self.args.yita) * self.mae2(y_pred[..., 1], y_true[..., 1], pe_weights)
            loss2 = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
                    (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
            
            loss = loss1 + loss2
        else:
            loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0]) + \
                    (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss
    
    def aleatoric_uncertainty(self, x0, graph):

        delta_x_in = torch.matmul(graph.unsqueeze(0).unsqueeze(0), self.PE.x_in)

        delta_x_out = torch.matmul(graph.unsqueeze(0).unsqueeze(0), self.PE.x_out)
        
        k = torch.pow((self.z_in - delta_x_out.reshape(self.args.batch_size, 1, len(graph))) + (self.z_out - delta_x_in.reshape(self.args.batch_size, 1, len(graph))), 2)

        return k
