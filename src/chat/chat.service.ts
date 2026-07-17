import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model, Types } from 'mongoose';
import { ChatSession } from './schemas/chat-session.schema';
import { ChatMessage } from './schemas/chat-message.schema';

@Injectable()
export class ChatService {
  constructor(
    @InjectModel(ChatSession.name) private readonly sessionModel: Model<ChatSession>,
    @InjectModel(ChatMessage.name) private readonly messageModel: Model<ChatMessage>,
  ) { }

  // Create a new session
  async createSession(userId: string, firstQuestion: string): Promise<ChatSession> {
    const subject = firstQuestion.length > 50
      ? firstQuestion.substring(0, 50) + '...'
      : firstQuestion;

    const session = new this.sessionModel({ userId, subject });
    return session.save();
  }

  private getThreeMonthsAgo() {
    const date = new Date();
    date.setMonth(date.getMonth() - 3);
    return date;
  }

  // Save a question/answer message to a session
  async addMessage(sessionId: string, role: 'user' | 'assistant', content: string): Promise<ChatMessage> {
    const sessionExists = await this.sessionModel.exists({ _id: new Types.ObjectId(sessionId) });
    if (!sessionExists) {
      throw new NotFoundException(`Chat session not found for ID: ${sessionId}`);
    }

    const message = new this.messageModel({
      sessionId: new Types.ObjectId(sessionId),
      role,
      content,
    });

    // Update the updated_at timestamp on the session to bubble it up in lists
    await this.sessionModel.updateOne(
      { _id: new Types.ObjectId(sessionId) },
      { $set: { updatedAt: new Date() } }
    );

    return message.save();
  }

  // Get user session list
  async getUserSessions(userId: string): Promise<ChatSession[]> {
    return this.sessionModel
      .find({ userId, createdAt: { $gte: this.getThreeMonthsAgo() } })
      .sort({ createdAt: -1 })
      .exec();
  }

  async getUserSessionHistory(userId: string) {

    const sessions = await this.sessionModel
      .find({ userId, createdAt: { $gte: this.getThreeMonthsAgo() } })
      .select('_id subject createdAt updatedAt')
      .sort({ createdAt: -1 })
      .lean()
      .exec();

    return sessions.map((session: any) => ({
      sessionId: session._id.toString(),
      subject: session.subject,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
    }));
  }

  // Get conversation history in order
  async getSessionHistory(sessionId: string): Promise<ChatMessage[]> {
    return this.messageModel.find({ sessionId: new Types.ObjectId(sessionId) })
      .sort({ createdAt: 1 })
      .exec();
  }

  // Delete a session and its associated messages
  async deleteSession(sessionId: string): Promise<any> {
    const id = new Types.ObjectId(sessionId);
    await this.messageModel.deleteMany({ sessionId: id }).exec();
    return this.sessionModel.deleteOne({ _id: id }).exec();
  }
}
