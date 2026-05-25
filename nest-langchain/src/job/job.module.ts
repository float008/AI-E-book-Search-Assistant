import { Module, forwardRef } from '@nestjs/common';
import { JobAgentService } from '../ai/job-agent.service';
import { ToolModule } from '../tool/tool.module';
import { JobService } from './job.service';

@Module({
  imports: [forwardRef(() => ToolModule)],
  providers: [JobService, JobAgentService],
  exports: [JobService],
})
export class JobModule {}
