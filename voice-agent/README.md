# LiveKit Mock Interview Agent

A multi-agent voice-based interview system built with LiveKit that conducts mock interviews for candidates. The system features intelligent agent handoffs, persistent conversation context, and dynamic agent routing based on interview stage.

## Overview

The Mock Interview Agent is a sophisticated multi-agent system that simulates a professional interview experience. It uses LiveKit's real-time communication capabilities combined with multiple AI services for speech-to-text, language understanding, and text-to-speech.

### Key Features

- **Multi-Agent Architecture**: Seamlessly transitions between specialized agents during the interview
- **Persistent Context**: Maintains conversation history and candidate information across agent handoffs
- **Real-Time Communication**: Built on LiveKit for low-latency voice interaction
- **Advanced Speech Processing**: 
  - STT: Deepgram Nova-3 for accurate speech recognition
  - TTS: Cartesia Sonic-3 for natural voice synthesis
  - VAD: Silero Voice Activity Detection for intelligent conversation flow
- **Flexible LLM Integration**: OpenAI GPT-4-mini for intelligent responses
- **Noise Cancellation**: Optional Krisp BVC noise cancellation (Linux/macOS support)

## Architecture

### Agent Flow

```
IntroAgent → Prev_experience_Agent → [interview_finished]
```

#### IntroAgent
- **Purpose**: Initial greeting and candidate introduction
- **Responsibilities**:
  - Welcome the candidate
  - Collect candidate name and basic introduction
  - Trigger handoff to experience agent via `information_gathered()` function tool

#### Prev_experience_Agent
- **Purpose**: Gather candidate's professional background
- **Responsibilities**:
  - Ask about previous work experiences
  - Inquire about internships or full-time roles
  - Conclude the interview with `interview_finished()` function tool
  - Generate personalized goodbye message

### Shared Data Structure

```python
@dataclass
class InterviewData:
    name: str | None = None
    prev_org: str | None = None
    prev_role: str | None = None
    exp: str | None = None
```

This dataclass maintains state across agent transitions and is passed to function tools.

## Setup Instructions

### Prerequisites

- Python 3.8+
- LiveKit server instance
- API credentials for:
  - OpenAI (GPT-4-mini)
  - Deepgram (Nova-3 STT)
  - Cartesia (Sonic-3 TTS)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dakshigoel22/mock_interview_agent.git
   cd mock_interview_agent/voice-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env.local` file in the voice-agent directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   DEEPGRAM_API_KEY=your_deepgram_api_key
   CARTESIA_API_KEY=your_cartesia_api_key
   LIVEKIT_URL=your_livekit_server_url
   LIVEKIT_API_KEY=your_livekit_api_key
   LIVEKIT_API_SECRET=your_livekit_api_secret
   ```

### Running the Agent

```bash
python agent.py
```

The agent will start the LiveKit agent server and listen for incoming interview sessions.

## API Services Used

### Speech-to-Text (STT)
- **Provider**: Deepgram
- **Model**: nova-3:multi
- **Features**: Multi-language support, real-time transcription

### Text-to-Speech (TTS)
- **Provider**: Cartesia
- **Model**: Sonic-3
- **ID**: 9626c31c-bec5-4cca-baa8-f8ba9e84c8bc
- **Features**: Natural voice synthesis with low latency

### Language Model
- **Provider**: OpenAI
- **Model**: gpt-4.1-mini
- **Features**: Fast, cost-effective responses with GPT-4 quality

### Voice Activity Detection (VAD)
- **Provider**: Silero
- **Features**: Accurate silence detection for conversation management

## Function Tools

### IntroAgent

#### `information_gathered(name: str, exp: str)`
Called when the candidate has provided their introduction.

**Parameters**:
- `name`: Candidate's full name
- `exp`: Candidate's experience description

**Behavior**: Stores candidate information and transitions to `Prev_experience_Agent`

### Prev_experience_Agent

#### `interview_finished()`
Signals the end of the interview session.

**Behavior**:
- Generates a personalized goodbye message
- Informs candidate about callback eligibility
- Closes the LiveKit room

## Conversation Instructions

The agents follow specific instructions to ensure professional, structured interviews:

**Common Instructions**:
- Act as "Kiya, a senior AI developer at Jobnova"
- Maintain formal and polite tone
- Do not repeat candidate statements
- Do not answer candidate questions (interviewer role)
- Do not use existing knowledge (focus on candidate's background)

**IntroAgent Instructions**:
- Greet the candidate
- Request name and introduction

**Prev_experience_Agent Instructions**:
- Ask about past work experiences
- Inquire about internships or full-time positions
- Avoid follow-up questions
- Call `interview_finished()` when satisfied

## Metrics and Monitoring

The system collects and logs usage metrics including:
- Token usage
- API call counts
- Session duration
- Performance metrics

Access logs through the `UsageCollector`:
```python
usage_collector = metrics.UsageCollector()
summary = usage_collector.get_summary()
logger.info(f"Usage: {summary}")
```

## Configuration

### Optional Features

#### Noise Cancellation
Uncomment the import to enable Krisp BVC:
```python
from livekit.plugins import noise_cancellation
```
**Note**: Currently supported on Linux and macOS only.

### Customization

Modify agent instructions by editing the `common_instructions` variable and individual agent constructors to change:
- Company and role names
- Interview questions and topics
- Conversation flow
- Model parameters

## Extending the System

To add additional agents:

1. Create a new agent class inheriting from `Agent`
2. Define custom instructions
3. Implement `on_enter()` method
4. Add function tools for agent transitions
5. Return the next agent from function tools to enable handoffs

Example:
```python
class TechnicalAgent(Agent):
    def __init__(self, name: str, *, chat_ctx: ChatContext | None = None):
        super().__init__(
            instructions=f"Technical interview instructions for {name}",
            chat_ctx=chat_ctx,
        )
    
    async def on_enter(self):
        self.session.generate_reply()
```

## Dependencies

Key dependencies:
- `livekit`: Core LiveKit SDK
- `livekit-agents`: Agent framework
- `livekit-plugins-deepgram`: Speech-to-text
- `livekit-plugins-openai`: Language model and TTS
- `livekit-plugins-silero`: Voice activity detection
- `python-dotenv`: Environment variable management

See `requirements.txt` for complete list.

## Logging

The system uses Python's standard logging module:

```python
logger = logging.getLogger("multi-agent")
logger.info("switching to the Experience agent with the provided user data: %s", context.userdata)
```

Set logging level in your configuration:
```python
logging.basicConfig(level=logging.INFO)
```

## Troubleshooting

### Agent Not Responding
- Verify API keys are correctly set in `.env.local`
- Check LiveKit server connectivity
- Ensure sufficient API quota on OpenAI, Deepgram, and Cartesia

### Poor Audio Quality
- Check internet connection stability
- Verify microphone input levels
- Consider enabling noise cancellation

### Agent Transition Issues
- Ensure function tools are properly called with required parameters
- Check that `ChatContext` is passed between agents
- Verify `context.userdata` is properly populated

## License

[Add your license here]

## Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.

## Support

For issues and questions, please open an issue on the GitHub repository.