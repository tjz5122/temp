import torch
import torch.nn as nn
import torch.nn.functional as F


###Mgnet
class MgIte(nn.Module): 
    def __init__(self, A, S):
        super().__init__()
        
        self.A = A
        self.S = S

        self.bn1 =nn.BatchNorm2d(A.weight.size(0)) 
        self.bn2 =nn.BatchNorm2d(S.weight.size(0)) 
    
    def forward(self, out):
        u, f = out 
        u = u + F.relu(self.bn2(self.S(F.relu(self.bn1((f-self.A(u))))))) 
        out = (u, f)
        return out



class MgRestriction(nn.Module):
    def __init__(self, A_old, A, Pi, R):
        super().__init__()

        self.A_old = A_old
        self.A = A
        self.Pi = Pi
        self.R = R

        self.bn1 = nn.BatchNorm2d(Pi.weight.size(0))   
        self.bn2 = nn.BatchNorm2d(R.weight.size(0))    

    def forward(self, out):
        u_old, f_old = out 
        u = F.relu(self.bn1(self.Pi(u_old)))                              
        f = F.relu(self.bn2(self.R(f_old-self.A_old(u_old)))) + self.A(u)        
        out = (u,f)
        return out




class MgNet(nn.Module):
    def __init__(self, num_channel_input, num_iteration, num_channel_u, num_channel_f, num_classes):
        super().__init__()
        self.num_iteration = num_iteration
        self.num_channel_u = num_channel_u
        self.conv1 = nn.Conv2d(num_channel_input, num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel_f)        

        
        A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=False)
        S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3,stride=1, padding=1, bias=False)
        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            for i in range(num_iteration_l):
                layers.append(MgIte(A, S)) 
            setattr(self, 'layer'+str(l), nn.Sequential(*layers))

            if l < len(num_iteration)-1:
                A_old = A 
                A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3,stride=1, padding=1, bias=False)
                S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3,stride=1, padding=1, bias=False)
                Pi = nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3,stride=2, padding=1, bias=False)
                R = nn.Conv2d(num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=1, bias=False)
                layers= [MgRestriction(A_old, A, Pi, R)] 
        
        self.pooling = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Linear(num_channel_u ,num_classes) 

    def forward(self, u, f):
        f = F.relu(self.bn1(self.conv1(f)))                
        use_cuda = torch.cuda.is_available()
        if use_cuda:                                        
            u = torch.zeros(f.size(0),self.num_channel_u,f.size(2),f.size(3), device=torch.device('cuda')) 
        else:
            u = torch.zeros(f.size(0),self.num_channel_u,f.size(2),f.size(3))        
        out = (u, f) 

        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer'+str(l))(out)
        u, f = out       
        u = self.pooling(u) #do avg pooling
        u = u.view(u.shape[0], -1)  #reshape u batch_Size to vector
        u = self.fc(u)
        return u
