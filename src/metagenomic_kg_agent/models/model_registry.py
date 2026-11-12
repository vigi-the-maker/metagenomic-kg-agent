from langchain_ollama import ChatOllama


class ModelRegistry:

    def __init__(self, config):
        self.config = config
        self.models = {}

    def get_model(self, agent_name):

        if agent_name not in self.config:
            raise ValueError(
                f"No model configuration found for: {agent_name}"
            )

        model_config = self.config[agent_name]

        provider = model_config["provider"]
        model_name = model_config["model"]
        temperature = model_config.get("temperature", 0)

        model_key = (
            provider,
            model_name,
            temperature
        )

        if model_key not in self.models:

            if provider == "ollama":
                model = ChatOllama(
                    model=model_name,
                    temperature=temperature
                )

            else:
                raise ValueError(
                    f"Unsupported model provider: {provider}"
                )

            self.models[model_key] = model

        return self.models[model_key]