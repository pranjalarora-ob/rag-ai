export interface ApiUserBy {
  id: string;
  name: string;
  customerId?: string;
  department?: string;
}

export interface ApiEpUser {
  userId(userId: any, question: string): unknown;
  id: string;
  name: string;
  emailId: string;
  epEmailId?: string;
  isAdmin: boolean;
  accountId: string;
  pimId: number;
  customerId: string;
  companyId: string;
}

export interface ApiCpUser {
  id: string;
  name: string;
  isAdmin: boolean;
  accountId: string;
  pimId: number;
  policies: [];
  departmentName?: string;
}

export enum PERMISSION_TYPE {
  VIEWER = 'Viewer',
  APPROVER = 'Approver',
  EDITOR = 'Editor',
  NA = 'NA',
}

export interface ObUser {
  id: string;
  name: string;
  emailId: string;
  imageUrl: string;
  isSuperAdmin: boolean;
  isAdmin: boolean;
  accountId: string;
  role: string;
}

