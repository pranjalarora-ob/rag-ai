import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Types } from 'mongoose';
import { ChatSession } from './chat-session.schema';

@Schema({ timestamps: true })
export class ChatMessage extends Document {
  @Prop({ type: Types.ObjectId, ref: 'ChatSession', required: true, index: true })
  sessionId: Types.ObjectId;

  @Prop({ required: true, enum: ['user', 'assistant'] })
  role: 'user' | 'assistant';

  @Prop({ required: true })
  content: string;
}

export const ChatMessageSchema = SchemaFactory.createForClass(ChatMessage);
