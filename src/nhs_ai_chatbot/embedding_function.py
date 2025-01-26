from langchain_aws import BedrockEmbeddings
import config


def get_embedding_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name=config.AWS_PROFILE, region_name=config.AWS_REGION
    )

    return embeddings
