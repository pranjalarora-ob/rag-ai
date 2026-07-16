import { createParamDecorator, ExecutionContext } from '@nestjs/common';

export const ApiUser = createParamDecorator((data: string, context: ExecutionContext) => {
  return context.switchToHttp().getRequest().user;
});

export const ApiCustomer = createParamDecorator((data: string, context: ExecutionContext) => {
  return context.switchToHttp().getRequest().customer;
});

export const ApiBy = createParamDecorator((data: string, context: ExecutionContext) => {
  const customer = context.switchToHttp().getRequest().customer;
  const user = context.switchToHttp().getRequest().user;

  if (customer && user) {
    return {
      id: user?.id,
      name: user?.name,
      epEmailId: user.emailId,
      emailId: customer.email,
      customerId: customer?.id,
      customerName: customer?.tradeName,
      companyId: customer?.companyId,
      department: user?.departmentName,
    };
  }

  if (customer) {
    return { customerId: customer?.id, emailId: customer.email, customerName: customer?.tradeName, companyId: customer?.companyId };
  }

  if (user) {
    return { id: user?.id, name: user?.name, epEmailId: user.emailId, department: user?.departmentName };
  }

  return undefined;
});
