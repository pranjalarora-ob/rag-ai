import { Injectable } from '@nestjs/common';

@Injectable()
export class GuardrailService {
  // Reject prompt-injection / rule-bypass attempts in the input.
  private readonly bannedInputPatterns = [
    /ignore (all )?previous instructions/i,
    /act as/i,
    /bypass/i,
    /reveal confidential/i,
    /jailbreak/i,
    /pretend to be/i,
    /override/i,
  ];

  private readonly sensitiveTerms = ['discount', 'refund', 'competitor', 'internal process'];

  isPolicyViolation(text: string): boolean {
    return this.bannedInputPatterns.some((regex) => regex.test(text));
  }

  containsSensitiveTerms(text: string): boolean {
    return this.sensitiveTerms.some((term) => text.toLowerCase().includes(term.toLowerCase()));
  }
}
