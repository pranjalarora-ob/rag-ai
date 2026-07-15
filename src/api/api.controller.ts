import { Controller } from '@nestjs/common';
import { ApiUserService } from './user.service';

@Controller()
export class ApiController {
  constructor(private readonly apiService: ApiUserService) {}
}
