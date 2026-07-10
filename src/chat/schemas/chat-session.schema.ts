import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document } from 'mongoose';

@Schema({ timestamps: true })
export class ChatSession extends Document {
  @Prop({ required: true, index: true })
  userId: string;

  @Prop({ required: true, default: 'New Conversation' })
  subject: string;
}

export const ChatSessionSchema = SchemaFactory.createForClass(ChatSession);
