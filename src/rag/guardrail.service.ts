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

  // Profanity / abuse (English + common Hindi). Word-boundaried and case-insensitive
  // to avoid false hits inside normal words (e.g. "class", "assessment").
  private readonly abusivePattern =
    /\b(f+u+c+k+\w*|f\*+k\w*|sh[i1]t\w*|b[i1]tch\w*|bastard\w*|assh?ole\w*|dickhead|dumbass|jackass|cunt\w*|slut\w*|whore\w*|motherf\w+|bullsh[i1]t|screw you|piss off|chut(iya|iye|iyapa)\w*|bhosdi\w*|bhosda\w*|m[au]d[ae]rch[o0]d\w*|b[ae]hench[o0]d\w*|gandu\w*|har[a]?mi\w*|rand[iy]\w*|saala|kamina|kutta|bsdk|mkc|bkl|ch0?du)\b/i;

  // Read-only assistant: it must NOT perform create/update/delete actions. Detects a
  // clear mutation command ("create a lead", "delete project", "assign owner") — an
  // action verb positioned BEFORE a domain object, so read queries like "project
  // schedule" or "how many projects changed stage" are not caught.
  private readonly actionVerb =
    /\b(create|add|register|update|edit|modify|delete|remove|cancel|approve|reject|assign|reassign|send|email|upload|import|raise|generate|convert|close)\b/i;
  private readonly domainObject =
    /\b(lead|project|task|milestone|user|account|customer|record|po|purchase order|boq|payment|invoice|quotation|meeting|reminder|note|comment|email|report|ticket)\b/i;

  isPolicyViolation(text: string): boolean {
    return this.bannedInputPatterns.some((regex) => regex.test(text));
  }

  isAbusive(text: string): boolean {
    return this.abusivePattern.test(text);
  }

  isOutOfScopeAction(text: string): boolean {
    const q = text.toLowerCase();
    const v = q.match(this.actionVerb);
    const o = q.match(this.domainObject);
    // Both present AND the verb comes before the object → it's a command to act.
    return !!v && !!o && q.indexOf(v[0]) <= q.indexOf(o[0]);
  }

  containsSensitiveTerms(text: string): boolean {
    return this.sensitiveTerms.some((term) => text.toLowerCase().includes(term.toLowerCase()));
  }
}
