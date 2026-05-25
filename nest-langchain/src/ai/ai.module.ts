import { Module } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiController } from './ai.controller';
import { ToolModule } from '../tool/tool.module';

@Module({
  imports: [ToolModule],
  controllers: [AiController],
  providers: [AiService],
})
export class AiModule {}
