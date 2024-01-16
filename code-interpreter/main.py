from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import AgentType

load_dotenv()


def main():
    print("start...")

    # model="gpt-4" 로 변경하면 훨씬 안정적으로 코드를 생성함
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    tool = PythonREPLTool()  # only one tool, PythonREPLTool
    python_agent_executor = create_python_agent(
        llm=llm,
        tool=tool,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    print(f"{tool=}")
    question_message = """
    generate and save in current working directory 15 QRcodes that point to url = "www.udemy.com/course/langchain". 
    TIPS: ALL Python libraries are already installed.
"""
    """
    필요한 패키지를 미리 다 설치해줘야 함.
    로컬 터미널을 실행하는 Tool은 없기 때문에 !!
    이후, 프롬프트로 라이브러리를 설치할 필요가 없다는 사실을 instruct 
        ( TIPS: ALL Python libraries are already installed. )
    """
    python_agent_executor.run(question_message)


if __name__ == "__main__":
    main()
