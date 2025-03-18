import flwr as fl

class CustomFedAvg(f1.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        print(f"Round {rnd} aggregated weights: {aggregated_weights}")
        return aggregated_weights
    
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=CustomFedAvg(),
)

