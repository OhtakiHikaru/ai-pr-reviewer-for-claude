import {info, warning} from '@actions/core'
import {AnthropicVertex} from '@anthropic-ai/vertex-sdk'
import pRetry from 'p-retry'
import {OpenAIOptions, Options} from './options'

// define type to save parentMessageId and conversationId
export interface Ids {
  parentMessageId?: string
  conversationId?: string
}

export class Bot {
  private readonly api: AnthropicVertex
  private readonly options: Options
  private readonly openaiOptions: OpenAIOptions

  constructor(options: Options, openaiOptions: OpenAIOptions) {
    this.options = options
    this.openaiOptions = openaiOptions
    this.api = new AnthropicVertex({
      region: options.vertexRegion,
      projectId: options.vertexProjectId
    })
  }

  chat = async (message: string, ids: Ids): Promise<[string, Ids]> => {
    let res: [string, Ids] = ['', {}]
    try {
      res = await this.chat_(message, ids)
      return res
    } catch (e: unknown) {
      warning(
        `Failed to chat: ${e}, backtrace: ${e instanceof Error ? e.stack : ''}`
      )
      return res
    }
  }

  private readonly chat_ = async (
    message: string,
    ids: Ids
  ): Promise<[string, Ids]> => {
    // record timing
    const start = Date.now()
    if (!message) {
      return ['', {}]
    }

    const currentDate = new Date().toISOString().split('T')[0]
    const systemMessage = `${this.options.systemMessage}
Current date: ${currentDate}

IMPORTANT: Entire response must be in the language with ISO code: ${this.options.language}
`

    let responseText = ''
    try {
      const response = await pRetry(
        () =>
          this.api.messages.create({
            model: this.openaiOptions.model,
            max_tokens: 4096,
            temperature: this.options.openaiModelTemperature,
            system: systemMessage,
            messages: [
              {
                role: 'user',
                content: message
              }
            ]
          }),
        {
          retries: this.options.openaiRetries
        }
      )

      const end = Date.now()
      info(`response: ${JSON.stringify(response)}`)
      info(`sendMessage (including retries) response time: ${end - start} ms`)

      if (response.content.length > 0 && response.content[0].type === 'text') {
        responseText = response.content[0].text
      }
    } catch (e: unknown) {
      info(
        `failed to send message to Claude: ${e}, backtrace: ${
          e instanceof Error ? e.stack : ''
        }`
      )
    }

    if (responseText === '') {
      warning('Claude response is empty')
    }

    // remove the prefix "with " in the response
    if (responseText.startsWith('with ')) {
      responseText = responseText.substring(5)
    }
    if (this.options.debug) {
      info(`Claude responses: ${responseText}`)
    }
    return [responseText, {}]
  }
}
