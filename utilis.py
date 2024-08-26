from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# prompt gerar folha de requisitos
def generate_requirements():
    prompt = ''' Contrua uma folha de requisitos com os principais pontos do texto transcrito abaixo, tenha em vista que quaisquer folha de requisito tem que ter um titulo com a principal funcionalidade (exemplo: Impressão prontuário – Inclusão atendimento ambulatorial), uma descrição (exemplo: Como usuário com perfil “imprime prontuário paciente” 
    Eu preciso que o sistema permita a impressão do prontuário do paciente com as informações do atendimento ambulatorial 
    Para permitir analise do prontuário do paciente impresso. ), um Use Store caso necessário, Critério(s) de Aceitação, cenario de aceite e Regras de apresentação. Estilize essa folha de requisitos em formato markdown com todos esses pontos
    
    conteúdo do treinamento fornecido:
    {contexto_docs}

    texto transcrito: {transcricao}
    
    Resposta:
    '''

    prompt_template = ChatPromptTemplate.from_template(prompt)

    return prompt_template

# prompt para armazenar respostas
def generate_responses():
    prompt = ''' Deixe armazenado as respostas a seguir para gerar a folha de requisitos posteriormente, retorne para o usuario que as 'respostas foram computadas!'

    respostas: {transcricao}
    '''

    prompt_template = ChatPromptTemplate.from_template(prompt)

    return prompt_template

# prompt da criação da ATA
def create_prompt_ata():
     prompt = ''' estruture está transcrição para que ela possa ser encaixada em um texto de ATA, lembre-se de retornar apenas a ATA, nenhuma outra mensagem
     
    transcrição: {transcricao}
    
    Resposta:
    '''

     prompt_template = ChatPromptTemplate.from_template(prompt)

     return prompt_template
 
# prompt das perguntas
def prompt_questions():
    prompt = ''' Dê sujestões de perguntas para completar uma folha de requisitos com base no texto transcrito abaixo, tenha em vista que quaisquer folha de requisito tem que ter um titulo com a principal funcionalidade (exemplo: Impressão prontuário – Inclusão atendimento ambulatorial), uma descrição (exemplo: Como usuário com perfil “imprime prontuário paciente” 
    Eu preciso que o sistema permita a impressão do prontuário do paciente com as informações do atendimento ambulatorial 
    Para permitir analise do prontuário do paciente impresso. ), um Use Store caso necessário, Critério(s) de Aceitação, cenario de aceite e Regras de apresentação.
    
    conteúdo do treinamento fornecido:
    {contexto_docs}
    
    Caso o contexto não tenha perguntas que agregem retorne 'Os principais pontos já estão sendo respondidos!'
    
    Retorne todas as perguntas que são necessarias para conclusão com '[Pergunta]:' no inicio da pergunta

    texto transcrito: {transcricao}
    
    Resposta:
    '''

    prompt_template = ChatPromptTemplate.from_template(prompt)

    return prompt_template

def processamento_text(type_prompt: str):
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    
    vectorstore_docs = PineconeVectorStore.from_existing_index(index_name='pns', embedding=embeddings)
    
    retriever_docs = vectorstore_docs.as_retriever()
    
    output_parser = StrOutputParser()
    modelo_llm = ChatOpenAI(model='gpt-4-turbo', temperature=0.5)
        
    if type_prompt == 'questions':
        prompt_ = prompt_questions()
        retreiver_geral = RunnableParallel(
            {
                'transcricao': RunnablePassthrough(), 
                "contexto_docs": retriever_docs
            }
        )
    elif type_prompt == 'requirement':
        prompt_ = generate_requirements()
        retreiver_geral = RunnableParallel(
        {
            'transcricao': RunnablePassthrough(), 
            "contexto_docs": retriever_docs
        }
    )
    elif type_prompt == 'responses':
        prompt_ = generate_responses()
        retreiver_geral = RunnableParallel(
        {
            'transcricao': RunnablePassthrough(), 
        })
    else:
        prompt_ = create_prompt_ata()
        retreiver_geral = RunnableParallel(
        {
            'transcricao': RunnablePassthrough(), 
        }
        )
        

    chain = (retreiver_geral | prompt_ | modelo_llm | output_parser)
    
    return chain