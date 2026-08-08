import torch

def count_parameters(model):
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total = sum(p.numel() for p in model.parameters())

    return {"trainable_parameters": trainable, "total_parameters": total}

def print_model_parameters(model):

    stats = count_parameters(model)

    print("====================")
    print("==Model Parameters==")
    print("====================")
    print(f"trainable: {stats ['trainable_parameters']:,}")
    print(f"total: {stats['total_parameters']:,}")

def print_training_stats(epoch, loss, branch_variance_ki, latent_uncertainty):
    print("=" * 40)
    print(f"Epoch:{epoch}")
    print(f"loss: {loss:.6f}")
    print(f"branch_variance - distance in form of ki:{branch_variance_ki.mean().item():.6f}")
    print(f"branch_variance - distiance in form of ki:{branch_variance_ki.max().item():.6f}")
    print(f"latent_uncertainty:{latent_uncertainty.mean().item():.6f}")
    print(f"latent_uncertainty:{latent_uncertainty.max().item():.6f}")
