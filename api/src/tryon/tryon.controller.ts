import {
  Controller,
  Get,
  HttpStatus,
  Logger,
  Post,
  Req,
  Res,
  UploadedFiles,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FileFieldsInterceptor } from '@nestjs/platform-express';
import { Request, Response } from 'express';
import { ConfigService } from '@nestjs/config';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import axios from 'axios';
import * as FormDataNS from 'form-data';
const FormData: any = (FormDataNS as any).default || FormDataNS;
import { TryonSession } from './tryon-session.entity';
import { OptionalJwtAuthGuard } from '../auth/optional-jwt.guard';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { ApiBearerAuth, ApiConsumes, ApiTags } from '@nestjs/swagger';

const FORWARDED_FIELDS = [
  'fit_scale',
  'alpha',
  'y_offset',
  'use_gen',
  'style_prompt',
  'gen_steps',
  'gen_guidance',
  'preserve_strength',
  'quality_preset',
  'refiner_mode',
  'cloth_type',
  'use_catvton_cloud',
];

@ApiTags('Try-On')
@Controller('tryon')
export class TryonController {
  private readonly logger = new Logger('Tryon');
  constructor(
    private cs: ConfigService,
    @InjectRepository(TryonSession) private sessions: Repository<TryonSession>,
  ) {}

  @ApiConsumes('multipart/form-data')
  @UseGuards(OptionalJwtAuthGuard)
  @Post()
  @UseInterceptors(
    FileFieldsInterceptor(
      [
        { name: 'person', maxCount: 1 },
        { name: 'cloth', maxCount: 1 },
      ],
      { limits: { fileSize: 25 * 1024 * 1024 } },
    ),
  )
  async tryon(
    @UploadedFiles()
    files: { person?: Express.Multer.File[]; cloth?: Express.Multer.File[] },
    @Req() req: Request,
    @Res() res: Response,
    @CurrentUser() user: any,
  ) {
    const person = files.person?.[0];
    const cloth = files.cloth?.[0];
    if (!person || !cloth) {
      return res.status(400).json({ message: 'Cần cả ảnh người và ảnh quần áo' });
    }

    const form = new FormData();
    form.append('person', person.buffer, {
      filename: person.originalname || 'person.jpg',
      contentType: person.mimetype,
    });
    form.append('cloth', cloth.buffer, {
      filename: cloth.originalname || 'cloth.jpg',
      contentType: cloth.mimetype,
    });
    for (const key of FORWARDED_FIELDS) {
      const v = (req.body as any)?.[key];
      if (v !== undefined && v !== null && v !== '') form.append(key, String(v));
    }

    const url = this.cs.get('VTON_URL', 'http://localhost:8000') + '/api/tryon';
    const timeout = +this.cs.get('VTON_TIMEOUT_MS', 1200000);

    try {
      const upstream = await axios.post(url, form, {
        headers: form.getHeaders(),
        responseType: 'stream',
        timeout,
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        validateStatus: () => true,
      });

      for (const h of ['x-pipeline-info', 'x-backend', 'x-warning', 'x-info']) {
        const v: any = upstream.headers[h];
        if (v) {
          const headerName = h.replace(/^x-/, 'X-').replace(/-(\w)/g, (_, c) => '-' + c.toUpperCase());
          res.setHeader(headerName, Array.isArray(v) ? v.join(', ') : String(v));
        }
      }
      res.status(upstream.status);
      res.setHeader('Content-Type', String(upstream.headers['content-type'] || 'image/png'));

      // Privacy: try-on images and session metadata are NOT persisted.
      // The result is streamed back to the user only; if they want to keep it,
      // they can download from the result panel on the frontend.

      upstream.data.pipe(res);
    } catch (err: any) {
      this.logger.error('VTON proxy error: ' + (err?.message || err));
      const code = err?.code === 'ECONNREFUSED' ? HttpStatus.BAD_GATEWAY : HttpStatus.GATEWAY_TIMEOUT;
      res.status(code).json({
        message: 'Không thể gọi tới dịch vụ Try-On (Flask :8000)',
        detail: err?.message,
      });
    }
  }

  @ApiBearerAuth('JWT')
  @UseGuards(JwtAuthGuard)
  @Get('history')
  history(@CurrentUser() u: any) {
    return this.sessions.find({
      where: { user: { id: u.id } },
      order: { createdAt: 'DESC' },
      take: 50,
    });
  }
}
