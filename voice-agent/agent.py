from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/dakshigoel/Desktop/mock_interview_agent/voice-agent/.env.local")

import logging
from dataclasses import dataclass

from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    metrics,
)
from livekit.agents.job import get_job_context
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram,openai, silero

# uncomment to enable Krisp BVC noise cancellation, currently supported on Linux and MacOS
from livekit.plugins import noise_cancellation

## The Mock-interview agent is a multi-agent that can handoff the session to another agent.
## Each agent could have its own instructions, as well as different STT, LLM, TTS,or realtime models.

logger = logging.getLogger("multi-agent")


common_instructions = (
    "You are Kiya, a senior AI developer at Jobnova. You are taking an interview for the candidate for the junior AI Developer internship role."
    "You have to be polite and formal. Do not use any existing knowledge. Do not repeat what the user mentions.DO NOT ANSWER ANY USER QUESTION AS YOU ARE AN INTERVIEWER."
)


@dataclass
class InterviewData:
    # Shared data that's used by the storyteller agent.
    # This structure is passed as a parameter to function calls.
    name: str | None = None
    prev_org: str| None = None
    prev_role: str| None = None
    exp : str | None = None 

class IntroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"{common_instructions}. Your goal is to start the interview. Greet the candidate."
            "You should ask the candidate for their name and an introduction."
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    @function_tool
    async def information_gathered(
        self,
        context: RunContext[InterviewData],
        name: str,
        exp: str
    ):
        """Called when the user has provided their introduction
        Args:
            name: The name of the user
        """

        context.userdata.name = name
        context.userdata.exp = exp

        # to carry through the current chat history, pass in the chat_ctx
        experience_agent = Prev_experience_Agent(name, chat_ctx=self.chat_ctx)

        logger.info(
            "switching to the Experience agent with the provided user data: %s", context.userdata
        )
        return experience_agent


class Prev_experience_Agent(Agent):
    "This agent will ask about the candidates previous experiences"
    def __init__(self, name: str, *, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=f"{common_instructions} The candidates's name is {name}."
            "You have to ask the candidate to tell about their past experiences and if they have done any internships or full-time."
            "Do not ask any follow-up questions"
            "Once you are satisfied and have completed the interview "
            "you MUST call the function `interview_finished`.",
            llm = "openai/gpt-4.1-mini",
            tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            chat_ctx=chat_ctx,
        )
        

    async def on_enter(self):
        # when the agent is added to the session, we'll initiate the conversation by
        # using the LLM to generate a reply
        self.session.generate_reply()
        
    @function_tool
    async def interview_finished(self, context: RunContext[InterviewData]):
        """When you are fininshed with the interview so call this function to end the conversation."""
        # interrupt any existing generation
        self.session.interrupt()
        # generate a goodbye message and hang up
        # awaiting it will ensure the message is played out before returning
        await self.session.generate_reply(
            instructions=f"Say thankyou and goodbye to {context.userdata.name} and let them know that if they are suitable then they will get a call back.", allow_interruptions=False
        )

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
    

    # @function_tool
    # async def previous_experience_gathered(
    #     self,
    #     context: RunContext[InterviewData],
    #     prev_org : str,
    #     prev_role : str
        
    # ):
    #     """
    #     Called when the user has provided their introduction
    #     Args:
    #         name: The name of the user
    #     """

    #     context.userdata.prev_org = prev_org
    #     context.userdata.prev_role = prev_role
        
    #     # # to carry through the current chat history, pass in the chat_ctx
    #     # interview_agent = Interview_Agent(name,prev_org, prev_role, chat_ctx=self.chat_ctx)

    #     # logger.info(
    #     #     "switching to the interview agent with the provided candidate's previous experiences: %s", context.userdata
    #     # )
    #     # return interview_agent
    
    # @function_tool
    # async def interview_finished(self, context: RunContext[InterviewData]):
    #     """When you are fininshed with the interview so call this function to end the conversation."""
    #     # interrupt any existing generation
    #     self.session.interrupt()
    #     # generate a goodbye message and hang up
    #     # awaiting it will ensure the message is played out before returning
    #     await self.session.generate_reply(
    #         instructions=f"Say thankyou and goodbye to {context.userdata.name} and let them know that if they are suitable then they will get a call back.", allow_interruptions=False
    #     )

    #     job_ctx = get_job_context()
    #     await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
    


# class Interview_Agent(Agent):
#     "This agent will ask about the candidates previous experiences"
#     def __init__(self, name: str,prev_org: str, prev_role: str, *, chat_ctx: ChatContext | None = None) -> None:
#         super().__init__(
#             instructions=f"{common_instructions} The candidates's name is {name}. The candidate has worked as {prev_role} at {prev_org}."
#             "Your goal is take an interview by asking a relevant questions about the candidates previous role/roles." 
#             "Ask small questions and do not reply to the user's questions"
#             # "Ask the first question and if the candidate is not able to answer, politely comfort and then move on to the next question."
#             # "Ask the second question andif the candidate is not able to answer, politely comfort the candidate."
#             "DO NOT tell any answers to the questions and stick to the candidates experiences that are relevant for the job role.",
#             ## each agent could override any of the model services, including mixing
#             ## realtime and non-realtime models
#             llm=openai.realtime.RealtimeModel(voice="echo"),
#             # llm=openai.LLM(model="gpt-4.1-mini"),
#             # tts=openai.TTS(voice="echo"),
#             tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
#             chat_ctx=chat_ctx,
#         )

#     async def on_enter(self):
#         # when the agent is added to the session, we'll initiate the conversation by
#         # using the LLM to generate a reply
#         self.session.generate_reply()

    # @function_tool
    # async def interview_finished(self, context: RunContext[InterviewData]):
    #     """When you are fininshed with the interview so call this function to end the conversation."""
    #     # interrupt any existing generation
    #     self.session.interrupt()
    #     # generate a goodbye message and hang up
    #     # awaiting it will ensure the message is played out before returning
    #     await self.session.generate_reply(
    #         instructions=f"Say thankyou and goodbye to {context.userdata.name} and let them know that if they are suitable then they will get a call back.", allow_interruptions=False
    #     )

    #     job_ctx = get_job_context()
    #     await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
    

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession[InterviewData](
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        # llm=openai.LLM(model="gpt-4.1-mini"),
        llm = "openai/gpt-4.1-mini",
        # stt=deepgram.STT(model="nova-3"),
        stt="deepgram/nova-3:multi",
        # tts=openai.TTS(voice="echo"),
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        userdata=InterviewData(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=IntroAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)