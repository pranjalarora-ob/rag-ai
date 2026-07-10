import { Controller, Get, Post, Delete, Body, Param, Query, NotFoundException } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiQuery, ApiBody } from '@nestjs/swagger';
import { ChatService } from './chat.service';

@ApiTags('chats')
@Controller('chats')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @ApiOperation({ summary: 'Create a new chat session' })
  @ApiBody({ schema: { properties: { userId: { type: 'string' }, firstQuestion: { type: 'string' } } } })
  @Post('session')
  async createSession(
    @Body('userId') userId: string,
    @Body('firstQuestion') firstQuestion: string,
  ) {
    return this.chatService.createSession(userId, firstQuestion);
  }

  @ApiOperation({ summary: 'Add a message to a session' })
  @ApiBody({ schema: { properties: { role: { type: 'string', enum: ['user', 'assistant'] }, content: { type: 'string' } } } })
  @Post('session/:sessionId/message')
  async addMessage(
    @Param('sessionId') sessionId: string,
    @Body('role') role: 'user' | 'assistant',
    @Body('content') content: string,
  ) {
    return this.chatService.addMessage(sessionId, role, content);
  }

  @ApiOperation({ summary: 'Get all chat sessions for a user' })
  @ApiQuery({ name: 'userId', type: 'string' })
  @Get()
  async getSessions(@Query('userId') userId: string) {
    return this.chatService.getUserSessions(userId);
  }

  @ApiOperation({ summary: 'Get the conversation history of a session' })
  @Get('session/:sessionId')
  async getHistory(@Param('sessionId') sessionId: string) {
    const history = await this.chatService.getSessionHistory(sessionId);
    if (!history) throw new NotFoundException('Session history not found');
    return history;
  }

  @ApiOperation({ summary: 'Delete a chat session' })
  @Delete('session/:sessionId')
  async deleteSession(@Param('sessionId') sessionId: string) {
    return this.chatService.deleteSession(sessionId);
  }
}
